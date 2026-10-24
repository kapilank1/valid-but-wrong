"""Physically consistent quadrotor simulator producing QUAD frames.

Conventions
    World  local NED (north, east, down), origin at the take-off point. The QUAD schema reports alt = -down and
           vu = -v_down, so up is positive for those two channels only.
    Body   FRD (forward, right, down). Attitude is ZYX Euler: R = Rz(yaw) Ry(pitch) Rx(roll) maps body -> NED.
    ax/ay/az  accelerometer specific force in body axes, f_body = R^T (a_ned - g_ned) with g_ned = [0, 0, +9.81].
           Level hover (or sitting on the ground) reads az ~ -9.81 m/s^2, the same sign ArduPilot's SCALED_IMU uses.
    p/q/r  body rates. Euler-angle rates follow from them through the kinematic matrix in euler_rates().
    yaw    wrapped to (-pi, pi]; consumers that difference it must unwrap.

Model (integrated at >= 200 Hz, decimated to rate_hz)
    * Translational: m v' = thrust + linear drag on airspeed (v - wind) + ground contact + m g; p' = v.
    * Rotational: body rates follow the commanded rates through a first-order lag (motor + inertia); Euler angles
      integrate the body rates through the exact kinematics. Thrust has its own first-order lag.
    * Wind: per-run mean (seeded direction and speed) plus Ornstein-Uhlenbeck gusts.
    * Ground: compliant spring-damper gear with friction, so touchdown and sitting on the pad stay consistent.
    * Controller: min-jerk waypoint reference -> PID position/velocity tracking -> desired specific force ->
      roll/pitch/thrust -> Euler-rate P loop -> body-rate commands. Classic cascaded multirotor structure.
    * Battery: 4S LiPo open-circuit curve vs state of charge, I*R sag with current ~ thrust^1.5, RC polarization.
    * Sensors: seeded white noise per channel, plus a slowly drifting (OU) barometer bias. groundspeed is
      hypot(vn, ve) of the *reported* velocities, as ArduPilot's VFR_HUD computes it.

Process noise (wind) and sensor noise use independent RNG streams derived from one seed, so the same seed always
produces the same flight and the same sensor readings.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from vbw.schema import QUAD, ChannelSchema, Frame
from vbw.sources.base import resolve_seed, seq_at, spawn_rngs

GRAVITY = 9.81
G_NED = np.array([0.0, 0.0, GRAVITY])
MIN_INTERNAL_HZ = 200.0

# Ground contact, per unit mass. The contact surface sits at alt = g / K so the vehicle rests at exactly alt = 0.
# Compliant gear (2.5 cm static compression) keeps the touchdown transient several output samples long.
GROUND_STIFFNESS = 400.0       # 1/s^2
GROUND_DAMPING = 40.0          # 1/s (critically damped with the stiffness above)
GROUND_FRICTION_MU = 0.8
GROUND_FRICTION_DAMPING = 20.0  # 1/s
CONTACT_ALT = GRAVITY / GROUND_STIFFNESS

YAW_RATE_PEAK = 0.6            # rad/s, peak of a min-jerk heading change
SPOOL_THRUST_FRACTION = 0.6    # of weight, reached at the end of the spool-up leg
GEAR_SETTLE_RATE = 3.0         # 1/s, how fast a disarmed vehicle on the pad levels out
MIN_JERK_PEAK_SPEED = 1.875    # peak of ds/dtau for the min-jerk profile


# --------------------------------------------------------------------------- vehicle parameters


@dataclass(frozen=True)
class Airframe:
    mass_kg: float = 1.6
    max_thrust_ratio: float = 2.2                              # max thrust / weight
    drag_per_s: tuple[float, float, float] = (0.30, 0.30, 0.45)  # linear drag accel per m/s of airspeed, NED axes
    tilt_max_rad: float = math.radians(30.0)
    rate_tau_s: float = 0.05
    thrust_tau_s: float = 0.04
    rate_max: tuple[float, float, float] = (3.0, 3.0, 1.5)    # rad/s


@dataclass(frozen=True)
class Gains:
    pos_h: float = 1.44        # natural freq 1.2 rad/s, damping 0.9
    vel_h: float = 2.16
    int_h: float = 0.25
    pos_v: float = 2.25        # natural freq 1.5 rad/s, damping 0.9
    vel_v: float = 2.7
    int_v: float = 0.4
    integral_limit: float = 4.0                     # m*s of accumulated position error, per axis
    attitude: tuple[float, float, float] = (6.0, 6.0, 2.5)  # Euler-rate command per rad of error


@dataclass(frozen=True)
class Battery:
    cells: int = 4
    capacity_ah: float = 5.0
    r_internal_ohm: float = 0.048
    r_polarization_ohm: float = 0.02
    tau_polarization_s: float = 30.0
    hover_current_a: float = 16.0
    avionics_current_a: float = 0.6
    initial_soc_range: tuple[float, float] = (0.92, 1.0)


# Resting LiPo cell voltage vs state of charge.
OCV_SOC = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
OCV_CELL_V = np.array([3.27, 3.61, 3.69, 3.73, 3.77, 3.79, 3.82, 3.87, 3.92, 3.98, 4.06, 4.20])


@dataclass(frozen=True)
class WindModel:
    mean_speed_range: tuple[float, float] = (2.0, 5.0)    # m/s, drawn once per run, direction uniform
    gust_sigma: tuple[float, float, float] = (1.0, 1.0, 0.3)  # m/s, OU stationary std per NED axis
    gust_tau_s: float = 4.0


# White sensor noise std per channel (EKF-output grade for position/velocity/attitude).
SENSOR_SIGMA: dict[str, float] = {
    "pn": 0.03, "pe": 0.03, "alt": 0.05,
    "vn": 0.03, "ve": 0.03, "vu": 0.04,
    "roll": 0.002, "pitch": 0.002, "yaw": 0.005,
    "p": 0.005, "q": 0.005, "r": 0.005,
    "ax": 0.08, "ay": 0.08, "az": 0.10,
    "battery_v": 0.015,
}
BARO_NOISE_M = 0.12
BARO_BIAS_SIGMA_M = 0.6
BARO_BIAS_TAU_S = 120.0


# --------------------------------------------------------------------------- missions


@dataclass(frozen=True)
class Leg:
    """One mission step: fly (min-jerk, rest to rest) to target, then dwell."""

    target: tuple[float, float, float]                 # (pn, pe, alt) m
    speed: float = 5.0                                 # peak speed along the leg, m/s
    hold_s: float = 0.0
    yaw: float | Literal["travel", "keep"] = "travel"  # rad, face the leg direction, or keep the current heading
    name: str | None = None                            # event label at the leg's start
    mode: Literal["fly", "spool", "disarmed"] = "fly"


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    p0: tuple[float, float, float]                     # NED
    p1: tuple[float, float, float]
    yaw0: float                                        # unwrapped
    yaw1: float
    yaw_turn_s: float
    mode: str


@dataclass(frozen=True)
class MissionPlan:
    name: str
    segments: tuple[Segment, ...]
    events: dict[str, float]
    wind: WindModel

    @property
    def duration_s(self) -> float:
        return self.segments[-1].t1


@dataclass(frozen=True)
class Reference:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: float
    yaw_rate: float
    mode: str
    progress: float                                    # 0..1 through the current segment


def landing_legs(descent_speed: float) -> list[Leg]:
    """Descend to 1.5 m, flare slowly to just below the pad (touchdown ~0.1 m/s), then disarm once on the ground."""
    pad = (0.0, 0.0, -0.05)
    return [
        Leg((0.0, 0.0, 1.5), speed=descent_speed, yaw="keep", name="land"),
        Leg(pad, speed=0.3, yaw="keep", name="flare"),
        Leg(pad, hold_s=4.0, yaw="keep", name="disarm", mode="disarmed"),
    ]


def _survey_legs() -> list[Leg]:
    bearing = math.radians(30.0)
    along = np.array([math.cos(bearing), math.sin(bearing)])
    across = np.array([-math.sin(bearing), math.cos(bearing)])
    corner = np.array([20.0, 10.0])
    alt, length, spacing, lines = 30.0, 90.0, 25.0, 4
    legs = [
        Leg((0.0, 0.0, 0.0), hold_s=3.0, yaw="keep", name="spool_up", mode="spool"),
        Leg((0.0, 0.0, alt), speed=3.0, yaw="keep", name="takeoff"),
        Leg((*corner, alt), speed=5.0, name="transit"),
    ]
    for i in range(lines):
        corner = corner + (length if i % 2 == 0 else -length) * along
        legs.append(Leg((*corner, alt), speed=7.0, name="survey" if i == 0 else None))
        if i < lines - 1:
            corner = corner + spacing * across
            legs.append(Leg((*corner, alt), speed=4.0))
    legs += [
        Leg((0.0, 0.0, alt), speed=8.0, name="return"),
        *landing_legs(descent_speed=2.0),
    ]
    return legs


def _delivery_legs() -> list[Leg]:
    drop = (160.0, 120.0)
    cruise_alt, drop_alt = 40.0, 8.0
    return [
        Leg((0.0, 0.0, 0.0), hold_s=3.0, yaw="keep", name="spool_up", mode="spool"),
        Leg((0.0, 0.0, cruise_alt), speed=3.0, yaw="keep", name="takeoff"),
        Leg((*drop, cruise_alt), speed=9.0, name="cruise"),
        Leg((*drop, drop_alt), speed=2.5, yaw="keep", name="descend"),
        Leg((*drop, drop_alt), hold_s=8.0, yaw="keep", name="drop"),
        Leg((*drop, cruise_alt), speed=3.0, yaw="keep", name="climb"),
        Leg((0.0, 0.0, cruise_alt), speed=9.0, name="return"),
        *landing_legs(descent_speed=2.5),
    ]


def _hover_legs() -> list[Leg]:
    alt = 12.0
    return [
        Leg((0.0, 0.0, 0.0), hold_s=3.0, yaw="keep", name="spool_up", mode="spool"),
        Leg((0.0, 0.0, alt), speed=2.0, yaw="keep", name="takeoff"),
        Leg((0.0, 0.0, alt), hold_s=35.0, yaw="keep", name="station_keep"),
        Leg((0.0, 0.0, alt), hold_s=20.0, yaw=1.2, name="yaw_scan"),
        Leg((0.0, 0.0, alt), hold_s=20.0, yaw=-0.9),
        Leg((0.0, 0.0, alt), hold_s=20.0, yaw=0.3),
        *landing_legs(descent_speed=1.5),
    ]


MISSIONS: dict[str, tuple[list[Leg], WindModel]] = {
    "survey": (_survey_legs(), WindModel(mean_speed_range=(2.0, 5.0), gust_sigma=(1.0, 1.0, 0.3))),
    "delivery": (_delivery_legs(), WindModel(mean_speed_range=(3.0, 6.0), gust_sigma=(1.2, 1.2, 0.3))),
    "hover": (_hover_legs(), WindModel(mean_speed_range=(2.0, 4.0), gust_sigma=(1.8, 1.8, 0.4))),
}


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    w = math.fmod(a + math.pi, 2.0 * math.pi)
    if w <= 0.0:
        w += 2.0 * math.pi
    return w - math.pi


def compile_plan(name: str, legs: list[Leg], wind: WindModel) -> MissionPlan:
    """Turn legs into timed min-jerk segments. Timing depends only on the plan, never on wind or seed."""
    segments: list[Segment] = []
    events: dict[str, float] = {}
    t, pos, yaw = 0.0, (0.0, 0.0, 0.0), 0.0
    for leg in legs:
        target = (leg.target[0], leg.target[1], -leg.target[2])  # (pn, pe, alt) -> NED
        if leg.name:
            events[leg.name] = round(t, 3)
        dn, de, dd = (target[i] - pos[i] for i in range(3))
        dist = math.sqrt(dn * dn + de * de + dd * dd)
        if leg.yaw == "keep":
            yaw_target = yaw
        elif leg.yaw == "travel":
            yaw_target = math.atan2(de, dn) if math.hypot(dn, de) > 5.0 else yaw
        else:
            yaw_target = float(leg.yaw)
        yaw1 = yaw + wrap_angle(yaw_target - yaw)
        yaw_turn = MIN_JERK_PEAK_SPEED * abs(yaw1 - yaw) / YAW_RATE_PEAK
        move = MIN_JERK_PEAK_SPEED * dist / leg.speed if dist > 0 else 0.0
        duration = max(move, yaw_turn, 1.0) if (dist > 0 or yaw_turn > 0) else 0.0
        if duration > 0:
            segments.append(Segment(t, t + duration, pos, target, yaw, yaw1, max(yaw_turn, 1.0), leg.mode))
            t += duration
        if leg.hold_s > 0:
            segments.append(Segment(t, t + leg.hold_s, target, target, yaw1, yaw1, 1.0, leg.mode))
            t += leg.hold_s
        pos, yaw = target, yaw1
    return MissionPlan(name, tuple(segments), events, wind)


@lru_cache(maxsize=None)
def mission_plan(mission: str) -> MissionPlan:
    if mission not in MISSIONS:
        raise KeyError(f"unknown quad mission {mission!r}; choose from {sorted(MISSIONS)}")
    legs, wind = MISSIONS[mission]
    return compile_plan(mission, legs, wind)


def min_jerk(tau: float) -> tuple[float, float, float]:
    """Shape s, ds/dtau, d2s/dtau2 of a rest-to-rest minimum-jerk move at normalized time tau."""
    tau = min(max(tau, 0.0), 1.0)
    s = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    ds = 30.0 * tau**2 * (1.0 - tau) ** 2
    dds = 60.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau)
    return s, ds, dds


def reference_at(plan: MissionPlan, starts: list[float], t: float) -> Reference:
    i = max(bisect.bisect_right(starts, t) - 1, 0)
    seg = plan.segments[i]
    duration = seg.t1 - seg.t0
    progress = min(max((t - seg.t0) / duration, 0.0), 1.0)
    s, ds, dds = min_jerk(progress)
    delta = np.subtract(seg.p1, seg.p0)
    pos = np.add(seg.p0, delta * s)
    vel = delta * (ds / duration)
    acc = delta * (dds / duration**2)
    ys, yds, _ = min_jerk((t - seg.t0) / seg.yaw_turn_s)
    yaw = seg.yaw0 + (seg.yaw1 - seg.yaw0) * ys
    yaw_rate = (seg.yaw1 - seg.yaw0) * yds / seg.yaw_turn_s
    return Reference(pos, vel, acc, yaw, yaw_rate, seg.mode, progress)


# --------------------------------------------------------------------------- kinematics


def rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def euler_rates(roll: float, pitch: float, body_rates: np.ndarray) -> np.ndarray:
    """ZYX Euler-angle rates from body rates p, q, r."""
    p, q, r = body_rates
    qr = q * math.sin(roll) + r * math.cos(roll)
    return np.array([
        p + math.tan(pitch) * qr,
        q * math.cos(roll) - r * math.sin(roll),
        qr / math.cos(pitch),
    ])


def body_rates_from_euler_rates(roll: float, pitch: float, euler_dot: np.ndarray) -> np.ndarray:
    """Inverse of euler_rates()."""
    roll_dot, pitch_dot, yaw_dot = euler_dot
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([
        roll_dot - sp * yaw_dot,
        cr * pitch_dot + sr * cp * yaw_dot,
        -sr * pitch_dot + cr * cp * yaw_dot,
    ])


def attitude_for_specific_force(f_ned: np.ndarray, yaw: float) -> tuple[float, float]:
    """Roll and pitch that point the thrust axis (body -z) along the desired specific force at a given yaw."""
    u = f_ned / np.linalg.norm(f_ned)
    forward = math.cos(yaw) * u[0] + math.sin(yaw) * u[1]
    right = -math.sin(yaw) * u[0] + math.cos(yaw) * u[1]
    roll = math.asin(min(max(right, -1.0), 1.0))
    pitch = math.atan2(-forward, -u[2])
    return roll, pitch


# --------------------------------------------------------------------------- forces and battery


def ground_contact(alt: float, vel: np.ndarray, other_force: np.ndarray) -> np.ndarray:
    """Specific force (NED) from the ground: spring-damper normal force plus friction that holds the skids."""
    penetration = CONTACT_ALT - alt
    if penetration <= 0.0:
        return np.zeros(3)
    normal = GROUND_STIFFNESS * penetration + GROUND_DAMPING * vel[2]  # vel[2] > 0 means moving down
    if normal <= 0.0:
        return np.zeros(3)
    friction = -other_force[:2] - GROUND_FRICTION_DAMPING * vel[:2]
    limit = GROUND_FRICTION_MU * normal
    magnitude = math.hypot(friction[0], friction[1])
    if magnitude > limit:
        friction *= limit / magnitude
    return np.array([friction[0], friction[1], -normal])


def open_circuit_voltage(soc: float, battery: Battery) -> float:
    return battery.cells * float(np.interp(soc, OCV_SOC, OCV_CELL_V))


def battery_current(thrust_n: float, weight_n: float, armed: bool, battery: Battery) -> float:
    """Momentum theory: rotor power ~ thrust^1.5."""
    if not armed:
        return battery.avionics_current_a
    return battery.avionics_current_a + battery.hover_current_a * (max(thrust_n, 0.0) / weight_n) ** 1.5


# --------------------------------------------------------------------------- simulator


@dataclass
class OrnsteinUhlenbeck:
    """Exact discretization of an OU process (stationary std sigma, correlation time tau), started stationary.
    Used for wind gusts and the barometer bias."""

    sigma: np.ndarray | float
    tau_s: float
    dt: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.decay = math.exp(-self.dt / self.tau_s)
        self.step_std = np.multiply(self.sigma, math.sqrt(1.0 - self.decay**2))
        self.value = self.rng.normal(0.0, self.sigma)

    def step(self) -> np.ndarray | float:
        self.value = self.value * self.decay + self.step_std * self.rng.standard_normal(np.shape(self.sigma))
        return self.value


@dataclass
class QuadState:
    pos: np.ndarray                     # NED, m
    vel: np.ndarray                     # NED, m/s
    euler: np.ndarray                   # roll, pitch, yaw (yaw unwrapped), rad
    rates: np.ndarray                   # body p, q, r, rad/s
    thrust_n: float = 0.0
    soc: float = 1.0
    v_polarization: float = 0.0
    integral: np.ndarray = field(default_factory=lambda: np.zeros(3))
    armed: bool = True


@dataclass
class QuadSim:
    """Source protocol implementation: a seeded quadrotor mission. frames() is repeatable for a given seed."""

    mission: str = "survey"
    seed: int | None = None
    rate_hz: float = 20.0
    duration_s: float | None = None
    seq_start: int = 0
    airframe: Airframe = field(default_factory=Airframe)
    gains: Gains = field(default_factory=Gains)
    battery: Battery = field(default_factory=Battery)
    schema: ChannelSchema = field(default=QUAD, init=False)
    vehicle_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.plan = mission_plan(self.mission)
        self.seed = resolve_seed(self.seed)
        if self.duration_s is None:
            self.duration_s = self.plan.duration_s
        self.vehicle_id = f"quad-{self.mission}"
        self.decimation = max(1, math.ceil(MIN_INTERNAL_HZ / self.rate_hz))
        self.internal_hz = self.rate_hz * self.decimation

    @property
    def events(self) -> dict[str, float]:
        return {k: v for k, v in self.plan.events.items() if v <= self.duration_s}

    # ---------------------------------------------------------------- public

    def frames(self) -> Iterator[Frame]:
        _, sensor_rng = spawn_rngs(self.seed, 2)
        baro_bias = OrnsteinUhlenbeck(BARO_BIAS_SIGMA_M, BARO_BIAS_TAU_S, 1.0 / self.rate_hz, sensor_rng)
        for i, (t, truth) in enumerate(self.truth_samples()):
            channels = measure(truth, sensor_rng, float(baro_bias.value))
            baro_bias.step()
            yield Frame(self.vehicle_id, QUAD.schema_id, seq_at(i, self.seq_start), t, channels)

    def truth_dataframe(self) -> pd.DataFrame:
        """Noise-free channels at rate_hz (plus wind and thrust for debugging)."""
        rows = [{"t": t, **truth} for t, truth in self.truth_samples(extras=True)]
        return pd.DataFrame(rows)

    def truth_samples(self, extras: bool = False) -> Iterator[tuple[float, dict[str, float]]]:
        """(t, true channels) at rate_hz. Re-runs the deterministic physics each call."""
        wind_rng, _ = spawn_rngs(self.seed, 2)
        af, bat = self.airframe, self.battery
        dt = 1.0 / self.internal_hz
        n_steps = int(round(self.duration_s * self.internal_hz))
        starts = [seg.t0 for seg in self.plan.segments]
        drag = np.array(af.drag_per_s)
        weight = af.mass_kg * GRAVITY

        heading = wind_rng.uniform(-math.pi, math.pi)
        speed = wind_rng.uniform(*self.plan.wind.mean_speed_range)
        wind_mean = np.array([speed * math.cos(heading), speed * math.sin(heading), 0.0])
        gusts = OrnsteinUhlenbeck(np.array(self.plan.wind.gust_sigma), self.plan.wind.gust_tau_s, dt, wind_rng)

        state = QuadState(
            pos=np.zeros(3), vel=np.zeros(3), euler=np.zeros(3), rates=np.zeros(3),
            soc=wind_rng.uniform(*bat.initial_soc_range),
        )

        for k in range(n_steps + 1):
            t = k * dt
            ref = reference_at(self.plan, starts, t)
            wind = wind_mean + gusts.value
            rates_cmd, thrust_cmd = self._control(state, ref, dt)

            roll, pitch, yaw = state.euler
            R = rotation_body_to_ned(roll, pitch, yaw)
            thrust_f = R[:, 2] * (-state.thrust_n / af.mass_kg)
            drag_f = -drag * (state.vel - wind)
            ground_f = ground_contact(-state.pos[2], state.vel, thrust_f + drag_f)
            specific_ned = thrust_f + drag_f + ground_f
            accel = specific_ned + G_NED

            current = battery_current(state.thrust_n, weight, state.armed, bat)
            if k % self.decimation == 0:
                battery_v = open_circuit_voltage(state.soc, bat) - current * bat.r_internal_ohm - state.v_polarization
                truth = true_channels(state, R.T @ specific_ned, battery_v)
                if extras:
                    truth.update(wind_n=wind[0], wind_e=wind[1], wind_d=wind[2], thrust_n=state.thrust_n, soc=state.soc)
                yield round(t, 9), truth

            # Integrate: velocity explicit, position trapezoidal (exact for the piecewise-linear velocity).
            vel_next = state.vel + accel * dt
            state.pos = state.pos + 0.5 * (state.vel + vel_next) * dt
            state.vel = vel_next
            state.euler = state.euler + euler_rates(roll, pitch, state.rates) * dt
            state.rates = state.rates + (rates_cmd - state.rates) * (dt / af.rate_tau_s)
            state.thrust_n += (thrust_cmd - state.thrust_n) * (dt / af.thrust_tau_s)
            state.soc -= current * dt / (bat.capacity_ah * 3600.0)
            state.v_polarization += (current * bat.r_polarization_ohm - state.v_polarization) * (
                dt / bat.tau_polarization_s
            )
            gusts.step()

    # ---------------------------------------------------------------- control

    def _control(self, state: QuadState, ref: Reference, dt: float) -> tuple[np.ndarray, float]:
        """Cascaded controller -> (body-rate command, thrust command in N)."""
        af, g = self.airframe, self.gains
        weight = af.mass_kg * GRAVITY
        roll, pitch, yaw = state.euler
        if ref.mode == "disarmed" and -state.pos[2] <= CONTACT_ALT:
            state.armed = False  # land detector: motors stop only once the gear is on the pad, then stay off
        if not state.armed:
            # Motors off: the landing gear settles the airframe level on the pad.
            state.integral[:] = 0.0
            settle = np.array([-GEAR_SETTLE_RATE * roll, -GEAR_SETTLE_RATE * pitch, 0.0])
            return body_rates_from_euler_rates(roll, pitch, settle), 0.0
        if ref.mode == "spool":
            state.integral[:] = 0.0
            thrust_cmd = SPOOL_THRUST_FRACTION * weight * ref.progress
            roll_des, pitch_des = 0.0, 0.0
        else:
            err_p = ref.pos - state.pos
            err_v = ref.vel - state.vel
            state.integral = np.clip(state.integral + err_p * dt, -g.integral_limit, g.integral_limit)
            kp = np.array([g.pos_h, g.pos_h, g.pos_v])
            kd = np.array([g.vel_h, g.vel_h, g.vel_v])
            ki = np.array([g.int_h, g.int_h, g.int_v])
            accel_cmd = ref.acc + kp * err_p + kd * err_v + ki * state.integral + np.array(af.drag_per_s) * ref.vel
            f_des = limit_tilt(accel_cmd - G_NED, af.tilt_max_rad)
            roll_des, pitch_des = attitude_for_specific_force(f_des, ref.yaw)
            thrust_cmd = min(af.mass_kg * float(np.linalg.norm(f_des)), af.max_thrust_ratio * weight)

        err_att = np.array([roll_des - roll, pitch_des - pitch, wrap_angle(ref.yaw - yaw)])
        euler_dot_cmd = np.array(g.attitude) * err_att + np.array([0.0, 0.0, ref.yaw_rate])
        rates_cmd = body_rates_from_euler_rates(roll, pitch, euler_dot_cmd)
        rate_max = np.array(af.rate_max)
        return np.clip(rates_cmd, -rate_max, rate_max), thrust_cmd


def limit_tilt(f_des: np.ndarray, tilt_max: float) -> np.ndarray:
    """Keep the desired specific force pointing up and within tilt_max of vertical, preserving its vertical part."""
    f = f_des.copy()
    f[2] = min(f[2], -0.3 * GRAVITY)  # always demand some upward thrust
    horizontal = math.hypot(f[0], f[1])
    limit = -f[2] * math.tan(tilt_max)
    if horizontal > limit:
        f[:2] *= limit / horizontal
    return f


def true_channels(state: QuadState, specific_body: np.ndarray, battery_v: float) -> dict[str, float]:
    alt = -float(state.pos[2])
    vn, ve = float(state.vel[0]), float(state.vel[1])
    return {
        "pn": float(state.pos[0]), "pe": float(state.pos[1]), "alt": alt,
        "vn": vn, "ve": ve, "vu": -float(state.vel[2]),
        "roll": float(state.euler[0]), "pitch": float(state.euler[1]), "yaw": wrap_angle(float(state.euler[2])),
        "p": float(state.rates[0]), "q": float(state.rates[1]), "r": float(state.rates[2]),
        "ax": float(specific_body[0]), "ay": float(specific_body[1]), "az": float(specific_body[2]),
        "baro_alt": alt,
        "groundspeed": math.hypot(vn, ve),
        "battery_v": float(battery_v),
    }


def measure(truth: dict[str, float], rng: np.random.Generator, baro_bias: float) -> dict[str, float]:
    """What the vehicle reports: truth + white noise, a biased barometer, groundspeed from reported velocity."""
    noisy = {name: truth[name] + rng.normal(0.0, sigma) for name, sigma in SENSOR_SIGMA.items()}
    noisy["yaw"] = wrap_angle(noisy["yaw"])
    noisy["baro_alt"] = truth["alt"] + baro_bias + rng.normal(0.0, BARO_NOISE_M)
    noisy["groundspeed"] = math.hypot(noisy["vn"], noisy["ve"])
    return {name: float(noisy[name]) for name in QUAD.channels}

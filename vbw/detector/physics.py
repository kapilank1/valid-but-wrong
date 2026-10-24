"""L1 physics-consistency rules.

Each rule is a pure, vectorized residual function ``residual(X, t, params) -> r`` over a window of reported frames
(``X`` is ``[..., T, C]`` in schema channel order, ``t`` is ``[..., T]``; ``r`` is ``[..., T]`` with NaN where the
window is too short). The same function serves three callers, so there is one implementation of each check:

* streaming: ``L1Checker`` keeps a ring buffer and evaluates every rule on its last frame,
* calibration: ``scripts/calibrate_l1.py`` runs the rules over whole clean launches,
* L3 features: ``vbw.detector.features`` runs them over batches of windows.

``evaluate_rule`` turns a residual into a ``Violation | None`` with a plain-language message.

Falcon 9 data is webcast OCR (1 Hz, upsampled to 10 Hz), so pointwise derivatives are noisy. The kinematic rules
therefore compare integrated quantities over a window (Δaltitude vs ∫vz dt) after a short median filter, and the
thresholds come from high percentiles of clean training launches (see calibrate_l1.py).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from vbw.schema import FALCON9, QUAD, ChannelSchema, Violation

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = ROOT / "models" / "l1_thresholds.json"

G0 = 9.80665
R_EARTH = 6_371_000.0


# --------------------------------------------------------------------------- array helpers (time on the last axis)


def _pad_front(x: np.ndarray, n: int) -> np.ndarray:
    """Left-pad the last axis with n NaNs so outputs line up with frame indices."""
    if n <= 0:
        return x
    pad = np.full(x.shape[:-1] + (n,), np.nan)
    return np.concatenate([pad, x], axis=-1)


def cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral along the last axis, starting at 0."""
    steps = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(t, axis=-1)
    zero = np.zeros(y.shape[:-1] + (1,))
    return np.concatenate([zero, np.cumsum(steps, axis=-1)], axis=-1)


def lagdiff(x: np.ndarray, lag: int) -> np.ndarray:
    """x[k] - x[k - lag], NaN for the first lag frames."""
    if x.shape[-1] <= lag:
        return np.full(x.shape, np.nan)
    return _pad_front(x[..., lag:] - x[..., :-lag], lag)


def rolling_median(x: np.ndarray, m: int) -> np.ndarray:
    """Trailing median over m frames (frame k uses k-m+1..k), NaN until m frames exist."""
    if m <= 1:
        return x
    if x.shape[-1] < m:
        return np.full(x.shape, np.nan)
    return _pad_front(np.median(sliding_window_view(x, m, axis=-1), axis=-1), m - 1)


def rolling_min(x: np.ndarray, m: int) -> np.ndarray:
    """Trailing minimum over m frames, NaN until m frames exist."""
    if x.shape[-1] < m:
        return np.full(x.shape, np.nan)
    return _pad_front(np.min(sliding_window_view(x, m, axis=-1), axis=-1), m - 1)


def integrated_mismatch(level: np.ndarray, rate: np.ndarray, t: np.ndarray, window: int, smooth: int) -> np.ndarray:
    """Average disagreement (level units / s) between Δlevel and ∫rate dt over `window` frames.

    level - ∫rate is constant when the two channels agree; a median over `smooth` frames makes single-sample OCR
    glitches harmless, then its change over the window divided by elapsed time is the rate mismatch.
    """
    offset = rolling_median(level - cumtrapz(rate, t), smooth)
    elapsed = lagdiff(t, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return lagdiff(offset, window) / elapsed


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def wrap_rad(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2 * math.pi) - math.pi


def frozen_run_length(X: np.ndarray, cap: int) -> np.ndarray:
    """Per frame and channel: how many consecutive frames (up to cap) the value has been bit-identical.

    Returns [..., T, C]. Only the last `cap` frames are looked at, so the result depends on a bounded window.
    """
    same = np.concatenate([np.zeros(X.shape[:-2] + (1, X.shape[-1]), bool), np.diff(X, axis=-2) == 0], axis=-2)
    run = np.zeros(X.shape, dtype=float)
    acc = np.zeros(X.shape[:-2] + (X.shape[-1],))
    for k in range(X.shape[-2]):
        acc = np.where(same[..., k, :], np.minimum(acc + 1, cap), 0.0)
        run[..., k, :] = acc
    return run


# --------------------------------------------------------------------------- rule definition


Residual = Callable[[np.ndarray, np.ndarray, dict], np.ndarray]
Message = Callable[[float, float, np.ndarray, dict], str]


@dataclass(frozen=True)
class Rule:
    """One physics check. `lookback` = frames of history needed before the newest frame."""

    name: str
    channels: tuple[str, ...]
    lookback: int
    residual: Residual
    unit: str
    message: Message
    default_threshold: float


def check_rule(rule: Rule, X: np.ndarray, t: np.ndarray, threshold: float, params: dict) -> tuple[float, Violation | None]:
    """Residual of the newest frame of a [T, C] window as a signed ratio to the threshold, plus the violation if any.

    Pure: same inputs, same answer. Ratio is 0 while the window is too short to evaluate the rule.
    """
    if X.shape[0] <= rule.lookback:
        return 0.0, None
    value = float(rule.residual(X, t, params)[-1])
    if not math.isfinite(value):
        return 0.0, None
    if abs(value) <= threshold:
        return value / threshold, None
    message = rule.message(value, threshold, X, params)
    return value / threshold, Violation(rule.name, list(rule.channels), round(value, 4), threshold, message)


def evaluate_rule(rule: Rule, X: np.ndarray, t: np.ndarray, threshold: float, params: dict) -> Violation | None:
    return check_rule(rule, X, t, threshold, params)[1]


# --------------------------------------------------------------------------- Falcon 9 rules


F = {c: i for i, c in enumerate(FALCON9.channels)}
F9_RATE_HZ = 10.0
F9_WINDOW = 100          # 10 s integration window
F9_LONG_WINDOW = 600     # 60 s: webcast altitude lag-then-catch-up glitches cancel, a sustained altitude lie doesn't
F9_SMOOTH = 10           # 1 s median before differencing
F9_MIN_SPEED = 50.0      # below this the flight-path angle is ill-defined
F9_MIN_FLIGHT_SPEED = 10.0
F9_MIN_Q_ALTITUDE = 200.0


def _f9(X: np.ndarray, name: str) -> np.ndarray:
    return X[..., F[name]]


def flight_path_angle_deg(vz: np.ndarray, vx: np.ndarray) -> np.ndarray:
    # The source defines angle against the horizontal regardless of direction (boostback has vx < 0).
    return np.degrees(np.arctan2(vz, np.abs(vx)))


def gravity(alt: np.ndarray) -> np.ndarray:
    return G0 * (R_EARTH / (R_EARTH + np.maximum(alt, -1000.0))) ** 2


def air_density(alt: np.ndarray, params: dict) -> np.ndarray:
    """Tabulated atmosphere fitted on training launches (log rho vs altitude), else a plain exponential."""
    atm = params.get("atmosphere")
    if atm:
        grid, log_rho = np.asarray(atm["alt"]), np.asarray(atm["log_rho"])
        rho = np.exp(np.interp(alt, grid, log_rho))
        return np.where(alt > grid[-1], 0.0, rho)
    return 1.225 * np.exp(-np.maximum(alt, 0.0) / 8500.0)


def f9_vz_vs_altitude(X, t, params):
    w, m = params.get("window", F9_WINDOW), params.get("smooth", F9_SMOOTH)
    return integrated_mismatch(_f9(X, "altitude"), _f9(X, "vz"), t, w, m)


def f9_vz_vs_altitude_long(X, t, params):
    """The same check over 60 s. Clean webcast data stays under ~14 m/s at this window (vs ~82 m/s at 10 s, where
    OCR lag-then-jump stretches dominate), so a slow altitude walk-off that hides in the 10 s noise shows up here."""
    return integrated_mismatch(_f9(X, "altitude"), _f9(X, "vz"), t, F9_LONG_WINDOW, params.get("smooth", F9_SMOOTH))


def f9_vx_vs_downrange(X, t, params):
    w, m = params.get("window", F9_WINDOW), params.get("smooth", F9_SMOOTH)
    return integrated_mismatch(_f9(X, "downrange"), _f9(X, "vx"), t, w, m)


def f9_accel_vs_speed(X, t, params):
    # Reported accel is sensed acceleration: d(speed)/dt + g·sin(γ). Remove gravity before integrating.
    alt, vz, vx = _f9(X, "altitude"), _f9(X, "vz"), _f9(X, "vx")
    gamma = np.arctan2(vz, np.abs(vx))
    kinematic = _f9(X, "accel") - gravity(alt) * np.sin(gamma)
    w, m = params.get("window", F9_WINDOW), params.get("smooth", F9_SMOOTH)
    mismatch = integrated_mismatch(_f9(X, "speed"), kinematic, t, w, m)
    # Standing on the pad or the landing zone the accel channel holds stale values; only judge it in flight.
    in_flight = rolling_min(_f9(X, "speed"), w + m) > F9_MIN_FLIGHT_SPEED
    return np.where(in_flight, mismatch, 0.0)


def f9_speed_vs_components(X, t, params):
    diff = np.hypot(_f9(X, "vx"), _f9(X, "vz")) - _f9(X, "speed")
    return rolling_median(diff, params.get("smooth", F9_SMOOTH))


def f9_angle_vs_velocity(X, t, params):
    diff = wrap_deg(_f9(X, "angle") - flight_path_angle_deg(_f9(X, "vz"), _f9(X, "vx")))
    diff = np.where(_f9(X, "speed") > F9_MIN_SPEED, diff, 0.0)
    return rolling_median(diff, params.get("smooth", F9_SMOOTH))


def predicted_q(X: np.ndarray, params: dict) -> np.ndarray:
    return 0.5 * air_density(_f9(X, "altitude"), params) * _f9(X, "speed") ** 2


def f9_q_vs_density(X, t, params):
    """Percent error of reported q vs ½ρ(alt)v², relative to max(q̂, floor) so vacuum (q≈0) stays well-defined."""
    q_hat = predicted_q(X, params)
    floor = params.get("q_floor", 1000.0)
    diff = 100.0 * (_f9(X, "q") - q_hat) / np.maximum(q_hat, floor)
    # Near the ground the OCR'd altitude reads 0 early (landing), and q there is small anyway.
    diff = np.where(_f9(X, "altitude") > F9_MIN_Q_ALTITUDE, diff, 0.0)
    return rolling_median(diff, params.get("q_smooth", 5))


def f9_accel_envelope(X, t, params):
    return np.abs(_f9(X, "accel"))


def f9_jerk_envelope(X, t, params):
    # Jerk over 1 s after a 0.5 s median; staging and MECO are in the clean calibration data, so their jerk is allowed.
    acc = rolling_median(_f9(X, "accel"), 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        return lagdiff(acc, 10) / lagdiff(t, 10)


def f9_corridor(X, t, params):
    """How far altitude/speed sit outside the nominal mission corridor at this mission time, in % of corridor scale.

    The corridor is the per-time-bin envelope of all training launches. It catches replayed old frames: the replayed
    state is self-consistent but belongs to a different mission time.
    """
    cor = params.get("corridor")
    if not cor:
        return np.zeros(t.shape)
    tg = np.asarray(cor["t"])
    worst = np.zeros(t.shape)
    # Lower bounds only hold while every training launch was still reporting; past that, launches end at different
    # times (landings reach 0 m) so only "too high / too fast" is checked.
    t_lo_end = cor.get("t_all_end", tg[-1])
    for ch in ("altitude", "speed"):
        lo = np.where(t <= t_lo_end, np.interp(t, tg, cor[f"{ch}_lo"]), -np.inf)
        hi = np.interp(t, tg, cor[f"{ch}_hi"], right=np.inf)
        scale = np.maximum(np.interp(t, tg, cor[f"{ch}_hi"]), cor[f"{ch}_floor"])
        x = _f9(X, ch)
        excess = np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)
        worst = np.maximum(worst, 100.0 * excess / scale)
    return worst


def f9_frozen_channel(X, t, params):
    """Longest bit-identical run (frames) among channels that should be changing. Real sensors always jitter."""
    run = np.moveaxis(frozen_run_length(X, params.get("frozen_cap", 50)), -1, -2)  # [..., C, T]
    moving = _f9(X, "speed") > 5.0
    q_live = predicted_q(X, params) > 50.0
    live = np.stack([moving] * len(FALCON9.channels), axis=-2)
    live[..., F["q"], :] &= q_live
    return np.max(np.where(live, run, 0.0), axis=-2)


def _msg_rate(what: str, unit: str) -> Message:
    def message(value, threshold, X, params):
        return f"{what} by {abs(value):.0f} {unit} (limit {threshold:.0f} {unit})"
    return message


def _msg_q(value, threshold, X, params):
    q = float(X[-1, F["q"]])
    q_hat = float(predicted_q(X[-1:], params)[0])
    return (f"Reported dynamic pressure {q / 1000:.1f} kPa disagrees with ½ρv² = {q_hat / 1000:.1f} kPa "
            f"by {abs(value):.0f}% (limit {threshold:.0f}%)")


def _msg_corridor(value, threshold, X, params):
    return (f"Reported altitude/speed is {value:.0f}% outside the nominal ascent corridor for this mission time "
            f"(limit {threshold:.0f}%)")


def _msg_frozen(value, threshold, X, params):
    run = frozen_run_length(X[-int(value) - 2:], int(value) + 1)[-1]
    names = [c for c, r in zip(FALCON9.channels, run) if r >= threshold]
    which = ", ".join(names) or "a channel"
    return f"Channel {which} frozen for {value / F9_RATE_HZ:.1f} s; real sensors never repeat exactly (limit {threshold / F9_RATE_HZ:.1f} s)"


FALCON9_RULES: tuple[Rule, ...] = (
    Rule("vz_vs_altitude_rate", ("vz", "altitude"), F9_WINDOW + F9_SMOOTH, f9_vz_vs_altitude, "m/s",
         _msg_rate("Reported vertical velocity disagrees with altitude rate", "m/s"), 90.0),
    Rule("vz_vs_altitude_rate_60s", ("vz", "altitude"), F9_LONG_WINDOW + F9_SMOOTH, f9_vz_vs_altitude_long, "m/s",
         _msg_rate("Over the last minute, altitude has drifted from integrated vertical velocity", "m/s"), 25.0),
    Rule("vx_vs_downrange_rate", ("vx", "downrange"), F9_WINDOW + F9_SMOOTH, f9_vx_vs_downrange, "m/s",
         _msg_rate("Reported horizontal velocity disagrees with downrange rate", "m/s"), 2.0),
    Rule("accel_vs_speed_rate", ("accel", "speed"), F9_WINDOW + F9_SMOOTH, f9_accel_vs_speed, "m/s²",
         _msg_rate("Reported acceleration (minus gravity) disagrees with speed rate", "m/s²"), 12.0),
    Rule("speed_vs_components", ("speed", "vx", "vz"), F9_SMOOTH, f9_speed_vs_components, "m/s",
         _msg_rate("Reported speed disagrees with √(vx² + vz²)", "m/s"), 45.0),
    Rule("angle_vs_velocity", ("angle", "vz", "vx"), F9_SMOOTH, f9_angle_vs_velocity, "deg",
         _msg_rate("Reported flight-path angle disagrees with the velocity vector", "deg"), 12.0),
    Rule("q_vs_density", ("q", "altitude", "speed"), 5, f9_q_vs_density, "%", _msg_q, 5.0),
    Rule("accel_envelope", ("accel",), 0, f9_accel_envelope, "m/s²",
         _msg_rate("Reported acceleration is beyond what Falcon 9 can do", "m/s²"), 60.0),
    Rule("jerk_envelope", ("accel",), 15, f9_jerk_envelope, "m/s³",
         _msg_rate("Reported acceleration changes faster than the engines can throttle", "m/s³"), 25.0),
    Rule("corridor", ("altitude", "speed"), 0, f9_corridor, "%", _msg_corridor, 25.0),
    Rule("frozen_channel", FALCON9.channels, 51, f9_frozen_channel, "frames", _msg_frozen, 20.0),
)


# --------------------------------------------------------------------------- quadrotor rules (local NED, alt/vu up)


Q = {c: i for i, c in enumerate(QUAD.channels)}
QUAD_RATE_HZ = 50.0
QUAD_WINDOW = 50         # 1 s
QUAD_SMOOTH = 5


def _q(X: np.ndarray, name: str) -> np.ndarray:
    return X[..., Q[name]]


def body_to_ned(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ZYX Euler rotation matrices, shape [..., 3, 3], body (FRD) -> NED."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.stack([
        np.stack([cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy], -1),
        np.stack([cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy], -1),
        np.stack([-sp, sr * cp, cr * cp], -1),
    ], -2)


def quad_ned_accel(X: np.ndarray) -> np.ndarray:
    """Inertial acceleration in NED from body specific force: a = R·f + g·ẑ. Returns [..., T, 3]."""
    rot = body_to_ned(_q(X, "roll"), _q(X, "pitch"), _q(X, "yaw"))
    f_body = np.stack([_q(X, "ax"), _q(X, "ay"), _q(X, "az")], -1)
    a = np.einsum("...ij,...j->...i", rot, f_body)
    a[..., 2] += G0
    return a


def euler_rates(X: np.ndarray) -> np.ndarray:
    """Euler angle rates from body rates (ZYX kinematics). Returns [..., T, 3]."""
    phi, theta = _q(X, "roll"), _q(X, "pitch")
    p, q, r = _q(X, "p"), _q(X, "q"), _q(X, "r")
    cos_t = np.where(np.abs(np.cos(theta)) < 1e-3, 1e-3, np.cos(theta))
    phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / cos_t
    return np.stack([phi_dot, theta_dot, psi_dot], -1)


def _worst_axis(parts: list[np.ndarray]) -> np.ndarray:
    """Signed value of the axis with the largest magnitude, per frame."""
    stacked = np.stack(parts, -1)
    idx = np.nanargmax(np.nan_to_num(np.abs(stacked), nan=-1.0), axis=-1)
    return np.take_along_axis(stacked, idx[..., None], -1)[..., 0]


def quad_pos_vs_vel(X, t, params):
    w, m = params.get("window", QUAD_WINDOW), params.get("smooth", QUAD_SMOOTH)
    return _worst_axis([integrated_mismatch(_q(X, p), _q(X, v), t, w, m)
                        for p, v in (("pn", "vn"), ("pe", "ve"), ("alt", "vu"))])


def quad_vel_vs_accel(X, t, params):
    w, m = params.get("window", QUAD_WINDOW), params.get("smooth", QUAD_SMOOTH)
    a = quad_ned_accel(X)
    up = -a[..., 2]  # vu is up, NED z is down
    return _worst_axis([integrated_mismatch(_q(X, "vn"), a[..., 0], t, w, m),
                        integrated_mismatch(_q(X, "ve"), a[..., 1], t, w, m),
                        integrated_mismatch(_q(X, "vu"), up, t, w, m)])


def quad_attitude_vs_rates(X, t, params):
    w, m = params.get("window", QUAD_WINDOW), params.get("smooth", QUAD_SMOOTH)
    rates = euler_rates(X)
    out = []
    for k, ang in enumerate(("roll", "pitch", "yaw")):
        # Unwrap so yaw crossing ±π doesn't look like a jump.
        level = np.unwrap(_q(X, ang), axis=-1)
        out.append(integrated_mismatch(level, rates[..., k], t, w, m))
    return np.degrees(_worst_axis(out))


def quad_baro_vs_gps(X, t, params):
    return rolling_median(_q(X, "baro_alt") - _q(X, "alt"), params.get("smooth", QUAD_SMOOTH))


def quad_groundspeed(X, t, params):
    return rolling_median(_q(X, "groundspeed") - np.hypot(_q(X, "vn"), _q(X, "ve")), params.get("smooth", QUAD_SMOOTH))


def quad_accel_envelope(X, t, params):
    return np.linalg.norm(np.stack([_q(X, "ax"), _q(X, "ay"), _q(X, "az")], -1), axis=-1)


def quad_frozen_channel(X, t, params):
    run = frozen_run_length(X, params.get("frozen_cap", 50))
    return np.max(run, axis=-1)


def _msg_frozen_quad(value, threshold, X, params):
    run = frozen_run_length(X[-int(value) - 2:], int(value) + 1)[-1]
    names = [c for c, r in zip(QUAD.channels, run) if r >= threshold]
    return f"Channel {', '.join(names) or 'a channel'} frozen for {int(value)} frames; real sensors never repeat exactly"


QUAD_RULES: tuple[Rule, ...] = (
    Rule("pos_vs_velocity", ("pn", "pe", "alt", "vn", "ve", "vu"), QUAD_WINDOW + QUAD_SMOOTH, quad_pos_vs_vel, "m/s",
         _msg_rate("Reported position change disagrees with integrated velocity", "m/s"), 1.0),
    Rule("velocity_vs_accel", ("vn", "ve", "vu", "ax", "ay", "az", "roll", "pitch", "yaw"), QUAD_WINDOW + QUAD_SMOOTH,
         quad_vel_vs_accel, "m/s²",
         _msg_rate("Reported velocity change disagrees with integrated accelerometer", "m/s²"), 2.0),
    Rule("attitude_vs_body_rates", ("roll", "pitch", "yaw", "p", "q", "r"), QUAD_WINDOW + QUAD_SMOOTH,
         quad_attitude_vs_rates, "deg/s",
         _msg_rate("Reported attitude change disagrees with integrated gyro rates", "deg/s"), 10.0),
    Rule("baro_vs_gps_alt", ("baro_alt", "alt"), QUAD_SMOOTH, quad_baro_vs_gps, "m",
         _msg_rate("Barometric altitude disagrees with GPS altitude", "m"), 5.0),
    Rule("groundspeed_vs_velocity", ("groundspeed", "vn", "ve"), QUAD_SMOOTH, quad_groundspeed, "m/s",
         _msg_rate("Reported groundspeed disagrees with √(vn² + ve²)", "m/s"), 1.0),
    Rule("accel_envelope", ("ax", "ay", "az"), 0, quad_accel_envelope, "m/s²",
         _msg_rate("Reported specific force is beyond the airframe's limits", "m/s²"), 40.0),
    Rule("frozen_channel", QUAD.channels, 51, quad_frozen_channel, "frames", _msg_frozen_quad, 25.0),
)


RULES_BY_SCHEMA: dict[str, tuple[Rule, ...]] = {"falcon9": FALCON9_RULES, "quad": QUAD_RULES}
NOMINAL_RATE_HZ: dict[str, float] = {"falcon9": F9_RATE_HZ, "quad": QUAD_RATE_HZ}


# --------------------------------------------------------------------------- calibrated config


@dataclass
class L1Config:
    """Per-schema thresholds and fitted parameters (atmosphere table, corridor), from models/l1_thresholds.json."""

    schema: str
    thresholds: dict[str, float]
    params: dict = field(default_factory=dict)
    calibrated: bool = False

    @classmethod
    def load(cls, schema: ChannelSchema, path: Path = THRESHOLDS_PATH) -> "L1Config":
        rules = RULES_BY_SCHEMA.get(schema.name, ())
        defaults = {r.name: r.default_threshold for r in rules}
        if path.exists():
            blob = json.loads(path.read_text()).get(schema.name)
            if blob:
                return cls(schema.name, {**defaults, **blob["thresholds"]}, blob.get("params", {}), True)
        return cls(schema.name, defaults, {}, False)


def max_lookback(schema: ChannelSchema) -> int:
    return max((r.lookback for r in RULES_BY_SCHEMA.get(schema.name, ())), default=0)


def rule_residual_series(schema: ChannelSchema, X: np.ndarray, t: np.ndarray, config: L1Config) -> dict[str, np.ndarray]:
    """All rule residuals over a whole sequence or a batch of windows ([..., T, C] -> {rule: [..., T]})."""
    return {r.name: r.residual(X, t, config.params) for r in RULES_BY_SCHEMA.get(schema.name, ())}


# --------------------------------------------------------------------------- streaming


class FrameBuffer:
    """Fixed-capacity ring buffer of frames that returns the last n frames as a contiguous [n, C] view."""

    def __init__(self, n_channels: int, capacity: int):
        self.capacity = capacity
        self._X = np.zeros((2 * capacity, n_channels))
        self._t = np.zeros(2 * capacity)
        self._end = 0

    def __len__(self) -> int:
        return min(self._end, self.capacity)

    def clear(self) -> None:
        self._end = 0

    def append(self, x: np.ndarray, t: float) -> None:
        if self._end == 2 * self.capacity:  # shift the newest half down; amortized O(1)
            self._X[: self.capacity] = self._X[self.capacity:]
            self._t[: self.capacity] = self._t[self.capacity:]
            self._end = self.capacity
        self._X[self._end] = x
        self._t[self._end] = t
        self._end += 1

    def last(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        n = min(n, len(self))
        return self._X[self._end - n: self._end], self._t[self._end - n: self._end]


@dataclass
class L1Result:
    violations: list[Violation]
    ratios: dict[str, float]          # signed residual / threshold per rule (0 when not yet evaluable)

    @property
    def max_ratio(self) -> float:
        return max((abs(r) for r in self.ratios.values()), default=0.0)


class L1Checker:
    """Streaming evaluation of every rule for one schema."""

    def __init__(self, schema: ChannelSchema, config: L1Config | None = None):
        self.schema = schema
        self.config = config or L1Config.load(schema)
        self.rules = RULES_BY_SCHEMA.get(schema.name, ())
        self.buffer = FrameBuffer(len(schema.channels), max_lookback(schema) + 1)

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, x: np.ndarray, t: float) -> L1Result:
        self.buffer.append(x, t)
        violations, ratios = [], {}
        for rule in self.rules:
            X, tt = self.buffer.last(rule.lookback + 1)
            ratio, violation = check_rule(rule, X, tt, self.config.thresholds[rule.name], self.config.params)
            ratios[rule.name] = ratio
            if violation is not None:
                violations.append(violation)
        violations.sort(key=lambda v: -abs(v.value) / v.threshold)
        return L1Result(violations, ratios)

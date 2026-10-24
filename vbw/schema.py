"""Shared data contract. Every module (sources, link, detector, attacker, pipeline, console) speaks these types.

Do not change field names without updating apps/console/src/lib/types.ts — the console mirrors this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator, Protocol


# --------------------------------------------------------------------------- channel schemas


@dataclass(frozen=True)
class ChannelSchema:
    """Ordered list of channels a vehicle reports. Order defines the wire layout."""

    schema_id: int
    name: str
    channels: tuple[str, ...]
    units: tuple[str, ...]

    def index(self, channel: str) -> int:
        return self.channels.index(channel)


# Falcon 9 ascent, derived from real webcast telemetry (shahar603/Telemetry-Data, "analysed" files).
# All SI units. vz = vertical velocity, vx = horizontal velocity, accel = d(speed)/dt as reported.
FALCON9 = ChannelSchema(
    schema_id=1,
    name="falcon9",
    channels=("speed", "altitude", "vz", "vx", "downrange", "accel", "angle", "q"),
    units=("m/s", "m", "m/s", "m/s", "m", "m/s^2", "deg", "Pa"),
)

# Multirotor drone (simulated quad or ArduPilot SITL via MAVLink). Local NED frame.
QUAD = ChannelSchema(
    schema_id=2,
    name="quad",
    channels=(
        "pn", "pe", "alt",          # position north/east (m), altitude up (m)
        "vn", "ve", "vu",           # velocity north/east/up (m/s)
        "roll", "pitch", "yaw",     # attitude (rad)
        "p", "q", "r",              # body rates (rad/s)
        "ax", "ay", "az",           # body specific force (m/s^2), az ~ -9.81 at hover
        "baro_alt",                 # barometric altitude (m), independent of GPS alt
        "groundspeed",              # horizontal speed (m/s)
        "battery_v",                # battery voltage (V)
    ),
    units=(
        "m", "m", "m",
        "m/s", "m/s", "m/s",
        "rad", "rad", "rad",
        "rad/s", "rad/s", "rad/s",
        "m/s^2", "m/s^2", "m/s^2",
        "m", "m/s", "V",
    ),
)

# NASA SMAP/MSL spacecraft telemetry (Hundman et al. 2018). One normalized telemetry value per channel;
# the 24 command one-hot columns are carried separately by the bench, not on the live link.
SPACECRAFT = ChannelSchema(
    schema_id=3,
    name="spacecraft",
    channels=("value",),
    units=("norm",),
)

SCHEMAS: dict[int, ChannelSchema] = {s.schema_id: s for s in (FALCON9, QUAD, SPACECRAFT)}
SCHEMAS_BY_NAME: dict[str, ChannelSchema] = {s.name: s for s in SCHEMAS.values()}


# --------------------------------------------------------------------------- frames & verdicts


@dataclass
class Frame:
    """One telemetry frame as the vehicle (or the attacker) claims it."""

    vehicle_id: str
    schema_id: int
    seq: int                     # uint16, wraps at 65536
    t: float                     # vehicle mission time, seconds
    channels: dict[str, float]

    def copy(self) -> "Frame":
        return Frame(self.vehicle_id, self.schema_id, self.seq, self.t, dict(self.channels))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class L0Result:
    """Link-layer integrity checks. A forged frame passes all of these — that's the point."""

    sync_ok: bool = True
    crc_ok: bool = True
    seq_ok: bool = True
    time_ok: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.sync_ok and self.crc_ok and self.seq_ok and self.time_ok


@dataclass
class Violation:
    """One physics-consistency rule that a frame broke."""

    rule: str                    # e.g. "vz_vs_dalt"
    channels: list[str]
    value: float                 # observed inconsistency (rule-specific units)
    threshold: float
    message: str                 # human readable, shown in console + spoken by voice callouts


@dataclass
class Verdict:
    vehicle_id: str
    seq: int
    t: float
    l0: L0Result
    l1_violations: list[Violation] = field(default_factory=list)
    l1_score: float = 0.0                        # 0..1
    l2_residuals: dict[str, float] = field(default_factory=dict)   # per-channel robust z-score
    l2_score: float = 0.0                        # 0..1
    l3_prob: float | None = None                 # hardened discriminator P(forged), None if not loaded
    score: float = 0.0                           # combined 0..1
    alert: bool = False
    explanation: str = ""
    detector: str = "base"                       # "base" | "hardened"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["l0"]["ok"] = self.l0.ok
        return d


# --------------------------------------------------------------------------- module interfaces


class Source(Protocol):
    """Produces ground-truth frames in mission-time order."""

    vehicle_id: str
    schema: ChannelSchema
    rate_hz: float

    def frames(self) -> Iterator[Frame]: ...


class Attack(Protocol):
    """Man-in-the-middle that rewrites decoded frames. The link layer re-encodes with a valid CRC."""

    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def apply(self, frame: Frame) -> Frame: ...


class Detector(Protocol):
    """Streaming detector. Stateful per vehicle; call reset() between scenarios."""

    name: str

    def reset(self) -> None: ...

    def update(self, frame: Frame, l0: L0Result) -> Verdict: ...

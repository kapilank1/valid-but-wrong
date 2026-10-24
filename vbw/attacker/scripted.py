"""Scripted man-in-the-middle attacks on the decoded telemetry stream.

Every attack implements the ``vbw.schema.Attack`` protocol: it is handed true frames in mission-time
order and returns the frame it wants the ground station to see. It never touches ``seq``, ``t``,
``vehicle_id`` or ``schema_id`` (the link layer re-encodes those with a fresh, valid CRC), so the forged
frame is indistinguishable from a real one at L0. The attack only begins once ``t >= start_t``; before
that it is the identity, and it is fully deterministic given a seed.

The interesting one is :class:`Coordinated`: it fakes a *different but kinematically self-consistent*
trajectory, so every naive per-channel physics rule (vz == d(alt)/dt, speed == hypot(vx, vz), ...) still
passes. That is the attack the learned detector layers are meant to catch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from vbw.schema import ChannelSchema, Frame

# ------------------------------------------------------------------ per-channel "unit lie" scales
# Magnitude of the lie at intensity == 1.0. BIAS is an absolute offset; DRIFT is a per-second rate.
_BIAS_SCALE: dict[str, dict[str, float]] = {
    "falcon9": {"altitude": 3000.0, "speed": 200.0, "vz": 100.0, "vx": 200.0,
                "downrange": 10000.0, "accel": 10.0, "angle": 20.0, "q": 4000.0},
    "quad": {"pn": 25.0, "pe": 25.0, "alt": 15.0, "vn": 5.0, "ve": 5.0, "vu": 5.0,
             "roll": 0.2, "pitch": 0.2, "yaw": 0.3, "p": 0.3, "q": 0.3, "r": 0.3,
             "ax": 2.0, "ay": 2.0, "az": 2.0, "baro_alt": 15.0, "groundspeed": 5.0, "battery_v": 1.0},
    "spacecraft": {"value": 0.8},
}
_DRIFT_SCALE: dict[str, dict[str, float]] = {  # per-second ramp rate at intensity == 1.0
    "falcon9": {"altitude": 30.0, "speed": 3.0, "vz": 2.0, "vx": 3.0,
                "downrange": 150.0, "accel": 0.3, "angle": 0.5, "q": 80.0},
    "quad": {"pn": 2.0, "pe": 2.0, "alt": 1.5, "vn": 0.5, "ve": 0.5, "vu": 0.5,
             "roll": 0.02, "pitch": 0.02, "yaw": 0.03, "p": 0.03, "q": 0.03, "r": 0.03,
             "ax": 0.2, "ay": 0.2, "az": 0.2, "baro_alt": 1.5, "groundspeed": 0.5, "battery_v": 0.05},
    "spacecraft": {"value": 0.08},
}
# The channel(s) a bias/drift attack walks off by default, per schema.
_DEFAULT_TARGET: dict[str, tuple[str, ...]] = {
    "falcon9": ("altitude",), "quad": ("alt",), "spacecraft": ("value",),
}
# Coordinated attack: position channel -> the velocity channel that must move with it, plus the
# constant velocity offset (m/s) at intensity == 1.0 that drives the slow walk-off.
_COORD: dict[str, dict] = {
    "falcon9": {"pos_vel": {"altitude": "vz"}, "ramp_rate": 25.0},
    "quad": {"pos_vel": {"alt": "vu"}, "ramp_rate": 3.0},
}


def _scale(table: dict[str, dict[str, float]], schema_name: str, channel: str) -> float:
    return table.get(schema_name, {}).get(channel, 1.0)


# ------------------------------------------------------------------ base


@dataclass
class _Attack:
    """Common bookkeeping: activation gate, seed, and a copy-on-apply helper."""

    schema: ChannelSchema
    start_t: float = 0.0
    intensity: float = 0.5
    name: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        self.intensity = float(np.clip(self.intensity, 0.0, 1.0))
        self._rng = np.random.default_rng(0)
        self._last: Frame | None = None

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(0 if seed is None else seed)
        self._last = None

    def _active(self, frame: Frame) -> bool:
        return frame.t >= self.start_t

    def apply(self, frame: Frame) -> Frame:  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, frame: Frame) -> Frame:
        out = frame.copy() if not self._active(frame) else self._apply_active(frame)
        self._last = frame.copy()
        return out

    def _apply_active(self, frame: Frame) -> Frame:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class NoneAttack(_Attack):
    """Identity. The control case: reported == truth."""

    name: ClassVar[str] = "none"

    def apply(self, frame: Frame) -> Frame:
        return frame.copy()

    def __call__(self, frame: Frame) -> Frame:
        return frame.copy()


@dataclass
class Bias(_Attack):
    """Constant additive offset on the target channel(s). L1 catches this almost immediately."""

    name: ClassVar[str] = "bias"
    channels: tuple[str, ...] = ()
    offset: float | None = None   # explicit absolute offset; otherwise intensity * per-channel scale

    def __post_init__(self) -> None:
        super().__post_init__()
        self.channels = tuple(self.channels) or _DEFAULT_TARGET.get(self.schema.name, (self.schema.channels[0],))

    def apply(self, frame: Frame) -> Frame:
        return self.__call__(frame)

    def _apply_active(self, frame: Frame) -> Frame:
        out = frame.copy()
        for ch in self.channels:
            off = self.offset if self.offset is not None else self.intensity * _scale(_BIAS_SCALE, self.schema.name, ch)
            out.channels[ch] = out.channels[ch] + off
        return out


@dataclass
class Drift(_Attack):
    """Slow linear ramp on the target channel(s): the classic valid-but-wrong walk-off."""

    name: ClassVar[str] = "drift"
    channels: tuple[str, ...] = ()
    rate: float | None = None     # explicit per-second rate; otherwise intensity * per-channel scale

    def __post_init__(self) -> None:
        super().__post_init__()
        self.channels = tuple(self.channels) or _DEFAULT_TARGET.get(self.schema.name, (self.schema.channels[0],))

    def apply(self, frame: Frame) -> Frame:
        return self.__call__(frame)

    def _apply_active(self, frame: Frame) -> Frame:
        out = frame.copy()
        tau = frame.t - self.start_t
        for ch in self.channels:
            rate = self.rate if self.rate is not None else self.intensity * _scale(_DRIFT_SCALE, self.schema.name, ch)
            out.channels[ch] = out.channels[ch] + rate * tau
        return out


@dataclass
class Replay(_Attack):
    """Record a window of earlier true frames and replay their channel values under the current
    seq/t. Each replayed frame is in-sequence and passes L0; the dynamics no longer match wall time."""

    name: ClassVar[str] = "replay"
    window_s: float = 6.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self._buffer: list[dict[str, float]] = []
        self._loop: list[dict[str, float]] | None = None
        self._i = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._buffer = []
        self._loop = None
        self._i = 0

    def apply(self, frame: Frame) -> Frame:
        return self.__call__(frame)

    def __call__(self, frame: Frame) -> Frame:
        if not self._active(frame):
            self._buffer.append(dict(frame.channels))
            # keep only the most recent window (10 Hz assumed for sizing; any rate still gives a loop)
            if len(self._buffer) > int(self.window_s * 10):
                self._buffer.pop(0)
            self._last = frame.copy()
            return frame.copy()
        if self._loop is None:
            self._loop = self._buffer or [dict(frame.channels)]
            self._i = 0
        out = frame.copy()
        out.channels = dict(self._loop[self._i % len(self._loop)])
        self._i += 1
        self._last = frame.copy()
        return out


@dataclass
class Freeze(_Attack):
    """Hold the last values seen before the attack began: a stuck sensor or frozen link."""

    name: ClassVar[str] = "freeze"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._held: dict[str, float] | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._held = None

    def apply(self, frame: Frame) -> Frame:
        return self.__call__(frame)

    def __call__(self, frame: Frame) -> Frame:
        if not self._active(frame):
            self._last = frame.copy()
            return frame.copy()
        if self._held is None:
            self._held = dict(self._last.channels) if self._last is not None else dict(frame.channels)
        out = frame.copy()
        out.channels = dict(self._held)
        self._last = frame.copy()
        return out


@dataclass
class Coordinated(_Attack):
    """A smart attacker that fakes a different but kinematically self-consistent trajectory.

    It walks a position channel off linearly and bumps the matching velocity channel by exactly the
    ramp rate, so vz == d(alt)/dt stays true, then recomputes every dependent channel (speed, angle,
    accel for falcon9; groundspeed, baro_alt for quad) so naive per-channel physics checks all pass.
    """

    name: ClassVar[str] = "coordinated"
    channels: tuple[str, ...] = ()   # position channels to target; default per schema

    def __post_init__(self) -> None:
        super().__post_init__()
        cfg = _COORD.get(self.schema.name)
        pos_vel = dict(cfg["pos_vel"]) if cfg else {}
        if self.channels:
            pos_vel = {c: pos_vel[c] for c in self.channels if c in pos_vel}
        self._pos_vel = pos_vel
        self._ramp = (cfg["ramp_rate"] if cfg else 0.0) * self.intensity
        self._last_speed: float | None = None
        self._last_t: float | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._last_speed = None
        self._last_t = None

    def apply(self, frame: Frame) -> Frame:
        return self.__call__(frame)

    def _apply_active(self, frame: Frame) -> Frame:
        out = frame.copy()
        tau = frame.t - self.start_t
        for pos_ch, vel_ch in self._pos_vel.items():
            out.channels[pos_ch] = out.channels[pos_ch] + self._ramp * tau
            out.channels[vel_ch] = out.channels[vel_ch] + self._ramp
        if self.schema.name == "falcon9":
            self._recompute_falcon9(out, frame)
        elif self.schema.name == "quad":
            self._recompute_quad(out, frame, tau)
        return out

    def _recompute_falcon9(self, out: Frame, frame: Frame) -> None:
        vx, vz = out.channels["vx"], out.channels["vz"]
        speed = math.hypot(vx, vz)
        out.channels["speed"] = speed
        out.channels["angle"] = math.degrees(math.atan2(vz, vx))
        if self._last_speed is not None and frame.t > (self._last_t or 0.0):
            out.channels["accel"] = (speed - self._last_speed) / (frame.t - self._last_t)
        self._last_speed, self._last_t = speed, frame.t

    def _recompute_quad(self, out: Frame, frame: Frame, tau: float) -> None:
        if "alt" in self._pos_vel:  # a smart attacker fakes the independent baro sensor too
            out.channels["baro_alt"] = out.channels["baro_alt"] + self._ramp * tau
        out.channels["groundspeed"] = math.hypot(out.channels["vn"], out.channels["ve"])


SCRIPTED: dict[str, type[_Attack]] = {
    cls.name: cls for cls in (NoneAttack, Bias, Drift, Replay, Freeze, Coordinated)
}

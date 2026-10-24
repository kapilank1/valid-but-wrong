"""Deterministic test doubles for the pipeline: a synthetic climb, a bias attack, a dead-reckoning detector.

Used by tests/test_pipeline_*.py so the runner, hub and server can be exercised without the real
source/attacker/detector modules. Never served as real data: `default_registries()` does not use them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator

from vbw.pipeline.registries import Registries
from vbw.schema import FALCON9, ChannelSchema, Frame, L0Result, Verdict, Violation

ACCEL = 10.0  # m/s^2, constant vertical acceleration of the synthetic climb


@dataclass
class ClimbSource:
    """Straight-up climb at constant acceleration in the FALCON9 schema."""

    n_frames: int = 600
    rate_hz: float = 100.0
    vehicle_id: str = "fake-1"
    schema: ChannelSchema = FALCON9

    def frames(self) -> Iterator[Frame]:
        for k in range(self.n_frames):
            t = k / self.rate_hz
            vz = ACCEL * t
            channels = {c: 0.0 for c in self.schema.channels}
            channels.update(speed=vz, altitude=0.5 * ACCEL * t * t, vz=vz, accel=ACCEL, angle=90.0)
            yield Frame(self.vehicle_id, self.schema.schema_id, k & 0xFFFF, t, channels)


@dataclass
class BiasAttack:
    """Adds `offset` metres to altitude from `start_t` on. `boom=True` makes apply() raise (error handling)."""

    name: str = "bias"
    start_t: float = 1.0
    offset: float = 50.0
    intensity: float = 1.0
    boom: bool = False
    seen: int = 0

    @property
    def params(self) -> dict[str, Any]:
        return {"start_t": self.start_t, "offset": self.offset, "intensity": self.intensity}

    def reset(self, seed: int | None = None) -> None:
        self.seen = 0

    def apply(self, frame: Frame) -> Frame:
        self.seen += 1
        if self.boom:
            raise RuntimeError("attack exploded")
        if self.name != "none" and frame.t >= self.start_t:
            frame.channels["altitude"] += self.offset * self.intensity
        return frame


@dataclass
class DeadReckoningDetector:
    """Alerts when altitude disagrees with the altitude obtained by integrating vz from the first frame."""

    name: str = "base"
    threshold: float = 5.0
    fail_every: int = 0                 # raise on every Nth update (0 = never)
    updates: int = 0
    seqs: list[int] = field(default_factory=list)
    _last: tuple[float, float] | None = None   # (t, vz)
    _alt_est: float = 0.0

    def reset(self) -> None:
        self.updates, self.seqs, self._last, self._alt_est = 0, [], None, 0.0

    def update(self, frame: Frame, l0: L0Result) -> Verdict:
        self.updates += 1
        self.seqs.append(frame.seq)
        if self.fail_every and self.updates % self.fail_every == 0:
            raise RuntimeError("detector hiccup")
        alt, vz = frame.channels["altitude"], frame.channels["vz"]
        if self._last is None:
            self._alt_est = alt
        else:
            t0, vz0 = self._last
            self._alt_est += 0.5 * (vz0 + vz) * (frame.t - t0)
        self._last = (frame.t, vz)
        residual = abs(alt - self._alt_est)
        alert = residual > self.threshold
        violations = [Violation("alt_vs_int_vz", ["altitude", "vz"], residual, self.threshold,
                                f"altitude off integrated vz by {residual:.1f} m")] if alert else []
        score = 1.0 - math.exp(-residual / self.threshold) if math.isfinite(self.threshold) else 0.0
        return Verdict(frame.vehicle_id, frame.seq, frame.t, l0, violations, l1_score=score, score=score,
                       alert=alert, explanation=violations[0].message if alert else "", detector=self.name)


class FakeWorld:
    """Fake registries that remember every object they build, for assertions."""

    SCENARIO = "fake:climb"

    def __init__(self, n_frames: int = 600, rate_hz: float = 100.0):
        self.n_frames, self.rate_hz = n_frames, rate_hz
        self.sources: list[ClimbSource] = []
        self.attacks: list[BiasAttack] = []
        self.detectors: list[DeadReckoningDetector] = []

    def list_scenarios(self) -> list[dict]:
        return [{"id": self.SCENARIO, "label": "Fake climb", "schema": "falcon9", "vehicle_id": "fake-1",
                 "rate_hz": self.rate_hz, "duration_s": self.n_frames / self.rate_hz, "events": {},
                 "description": "synthetic constant-acceleration climb"}]

    def make_source(self, scenario_id: str, seed: int | None = None) -> ClimbSource:
        if scenario_id != self.SCENARIO:
            raise KeyError(scenario_id)
        source = ClimbSource(self.n_frames, self.rate_hz)
        self.sources.append(source)
        return source

    def list_attacks(self) -> list[dict]:
        params = {"start_t": 1.0, "intensity": 1.0, "offset": 50.0}
        return [
            {"name": "none", "label": "None", "description": "", "schemas": ["falcon9"], "params": {}},
            {"name": "bias", "label": "Bias", "description": "", "schemas": ["falcon9"], "params": params},
            {"name": "boom", "label": "Broken", "description": "", "schemas": ["falcon9"], "params": {}},
            {"name": "quad_only", "label": "Quad", "description": "", "schemas": ["quad"], "params": {}},
        ]

    def make_attack(self, name: str, schema: ChannelSchema, **params: Any) -> BiasAttack:
        if name not in {a["name"] for a in self.list_attacks()}:
            raise KeyError(name)
        attack = BiasAttack(name=name, boom=(name == "boom"), **params)
        self.attacks.append(attack)
        return attack

    def list_detectors(self) -> list[dict]:
        return [{"name": n, "label": n, "description": ""} for n in ("base", "hardened", "blind", "flaky")]

    def make_detector(self, name: str, schema: ChannelSchema) -> DeadReckoningDetector:
        threshold = math.inf if name == "blind" else 5.0
        detector = DeadReckoningDetector(name=name, threshold=threshold, fail_every=10 if name == "flaky" else 0)
        self.detectors.append(detector)
        return detector

    def registries(self) -> Registries:
        return Registries(self.list_scenarios, self.make_source, self.list_attacks, self.make_attack,
                          self.list_detectors, self.make_detector)

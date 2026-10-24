"""Measure an attack against a detector: was it caught, how fast, and how big a lie it told first.

Used by both the PPO training report and ``scripts/make_forged_dataset.py`` (which writes
``results/attacks_eval.json``). Everything here is measurement only — no training, no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import numpy as np

from vbw.schema import ChannelSchema, Frame, L0Result

DetectorFactory = Callable[[], object]

# The channel whose lie defines the "operator picture error" per schema.
_PICTURE_CHANNEL = {"falcon9": "altitude", "quad": "alt", "spacecraft": "value"}


def frames_from_arrays(t: np.ndarray, x: np.ndarray, schema: ChannelSchema,
                       vehicle_id: str = "eval") -> Iterator[Frame]:
    for i in range(len(t)):
        channels = {c: float(x[i, j]) for j, c in enumerate(schema.channels)}
        yield Frame(vehicle_id, schema.schema_id, i & 0xFFFF, float(t[i]), channels)


@dataclass
class AttackResult:
    attack: str
    detected: bool
    latency_s: float | None          # mission-time from attack start to first alert
    max_lie_before_alert: float      # largest picture-channel lie up to first alert (or whole run)
    max_lie: float                   # largest picture-channel lie over the whole run
    mean_lie_before_alert: float     # mean lie over active frames before the first alert
    alert_rate: float                # fraction of frames that alerted
    frames: int


def evaluate_attack(
    t: np.ndarray,
    x: np.ndarray,
    schema: ChannelSchema,
    attack,
    detector,
    start_t: float,
    seed: int = 0,
) -> AttackResult:
    """Run one attack over one clean trajectory through one detector and measure the outcome."""
    attack.reset(seed)
    detector.reset()
    pic = _PICTURE_CHANNEL[schema.name]

    first_alert: int | None = None
    n_alert = 0
    max_lie = 0.0
    max_lie_before = 0.0
    active_lies: list[float] = []

    frames = list(frames_from_arrays(t, x, schema))
    for i, truth in enumerate(frames):
        reported = attack.apply(truth)
        verdict = detector.update(reported.copy(), L0Result())
        lie = abs(reported.channels[pic] - truth.channels[pic])
        max_lie = max(max_lie, lie)
        if truth.t >= start_t and first_alert is None:
            active_lies.append(lie)
            max_lie_before = max(max_lie_before, lie)
        if bool(getattr(verdict, "alert", False)):
            n_alert += 1
            if first_alert is None:
                first_alert = i

    detected = first_alert is not None
    latency = (float(t[first_alert]) - start_t) if detected else None
    mean_before = float(np.mean(active_lies)) if active_lies else 0.0
    return AttackResult(
        attack=getattr(attack, "name", "?"),
        detected=detected,
        latency_s=latency,
        max_lie_before_alert=max_lie_before,
        max_lie=max_lie,
        mean_lie_before_alert=mean_before,
        alert_rate=n_alert / max(len(frames), 1),
        frames=len(frames),
    )


def aggregate(results: list[AttackResult]) -> dict:
    """Mean over launches for one attack (None latencies dropped, detection rate reported instead)."""
    lat = [r.latency_s for r in results if r.latency_s is not None]
    return {
        "attack": results[0].attack,
        "runs": len(results),
        "detection_rate": float(np.mean([r.detected for r in results])),
        "mean_latency_s": float(np.mean(lat)) if lat else None,
        "mean_max_lie_before_alert": float(np.mean([r.max_lie_before_alert for r in results])),
        "mean_lie_before_alert": float(np.mean([r.mean_lie_before_alert for r in results])),
        "mean_max_lie": float(np.mean([r.max_lie for r in results])),
        "mean_alert_rate": float(np.mean([r.alert_rate for r in results])),
        "per_run": [asdict(r) for r in results],
    }

"""One scenario's telemetry link, frame by frame. Synchronous: no asyncio, no pacing.

    source.frames() --encode--> [vehicle downlink packet]
        --decode / attack.apply / encode--> [MITM packet, CRC recomputed so it is valid by construction]
        --LinkMonitor.receive--> (reported frame, L0)
        --detector.update--> Verdict

`truth` in a Tick is the frame as the vehicle transmitted it (decoded, so float32-quantized exactly like
`reported`); `attack_active` is true when the ground received something other than what was sent.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

from vbw.link.codec import LinkMonitor, decode, encode
from vbw.schema import Attack, ChannelSchema, Detector, Frame, L0Result, Source, Verdict

log = logging.getLogger("vbw.pipeline")

DEFAULT_WARM_FRAMES = 200


@dataclass
class Tick:
    truth: Frame
    reported: Frame
    l0: L0Result
    verdict: Verdict | None          # None only if the detector raised on this frame
    attack_active: bool
    errors: list[str] = field(default_factory=list)


def _same(a: float, b: float) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def frames_differ(a: Frame, b: Frame) -> bool:
    """True if the ground's picture differs from what the vehicle sent (values, seq, or time)."""
    if a.seq != b.seq or not _same(a.t, b.t) or a.channels.keys() != b.channels.keys():
        return True
    return any(not _same(a.channels[c], b.channels[c]) for c in a.channels)


def _err(stage: str, exc: Exception) -> str:
    return f"{stage}: {type(exc).__name__}: {exc}"


class Session:
    """Owns the per-run link state: source iterator, attack, ground link monitor, detector, recent history."""

    def __init__(
        self,
        source: Source,
        attack: Attack,
        detector: Detector,
        warm_frames: int = DEFAULT_WARM_FRAMES,
    ):
        self.source = source
        self.schema: ChannelSchema = source.schema
        self.vehicle_id: str = source.vehicle_id
        self.attack = attack
        self.detector = detector
        self.monitor = LinkMonitor(self.vehicle_id)
        # (truth, reported, l0) of the most recent frames; used to warm hot-swapped attacks/detectors.
        self.history: deque[tuple[Frame, Frame, L0Result]] = deque(maxlen=max(warm_frames, 0))
        self._frames: Iterator[Frame] = iter(source.frames())

    def next_truth(self) -> Frame | None:
        """Next ground-truth frame, or None when the source is exhausted."""
        return next(self._frames, None)

    def step(self, truth_in: Frame) -> Tick:
        errors: list[str] = []
        downlink = encode(truth_in)
        truth, _ = decode(downlink, self.vehicle_id)

        packet = downlink
        try:
            forged = self.attack.apply(truth.copy())
            packet = encode(forged)
        except Exception as exc:  # noqa: BLE001 - a broken attack must not take the link down
            errors.append(_err("attack", exc))

        reported, l0 = self.monitor.receive(packet)
        active = frames_differ(truth, reported)
        self.history.append((truth, reported, l0))

        verdict: Verdict | None = None
        try:
            verdict = self.detector.update(reported.copy(), _copy_l0(l0))
        except Exception as exc:  # noqa: BLE001
            errors.append(_err("detector", exc))
        return Tick(truth, reported, l0, verdict, active, errors)

    # ------------------------------------------------------------------ hot swap

    def warm_attack(self, attack: Attack) -> int:
        """Replay recent true frames through a new attack so stateful attacks (replay, PPO) have history."""
        warmed = 0
        for truth, _, _ in list(self.history):
            try:
                attack.apply(truth.copy())
                warmed += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("attack warm-up failed at seq %d: %s", truth.seq, exc)
                break
        return warmed

    def warm_detector(self, detector: Detector) -> int:
        """Replay recent reported frames (with their L0) so a hot-swapped detector does not start cold."""
        warmed = 0
        for _, reported, l0 in list(self.history):
            try:
                detector.update(reported.copy(), _copy_l0(l0))
                warmed += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("detector warm-up failed at seq %d: %s", reported.seq, exc)
                break
        return warmed

    def swap_attack(self, attack: Attack) -> int:
        warmed = self.warm_attack(attack)
        self.attack = attack
        return warmed

    def swap_detector(self, detector: Detector) -> int:
        warmed = self.warm_detector(detector)
        self.detector = detector
        return warmed


def _copy_l0(l0: L0Result) -> L0Result:
    return L0Result(l0.sync_ok, l0.crc_ok, l0.seq_ok, l0.time_ok, list(l0.reasons))

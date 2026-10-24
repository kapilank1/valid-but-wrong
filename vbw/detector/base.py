"""BaseDetector: L1 physics rules + L2 learned predictor, combined into one score with persistent alerting.

    det = BaseDetector(FALCON9)
    verdict = det.update(frame, l0)     # < 2 ms per frame on CPU

l1_score = soft(max rule residual / threshold); l2_score = soft(max(instant |z|, |EWMA z|) / thresholds);
score = noisy-OR of the two; alert = k-of-n persistence with hysteresis (alerting.py). Everything is 0.5 exactly at
its calibrated threshold, so τ = 0.5 means "some layer is at its clean-data limit".

If models/l2_<schema>.onnx is missing the detector runs L1-only (l2_score = 0); if l1_thresholds.json is missing
it uses each rule's built-in default threshold.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from vbw.detector.alerting import AlertConfig, AlertState, combine, soft_ratio
from vbw.detector.l2 import MODELS_DIR, L2Model, ewma_update, l2_ratio
from vbw.detector.physics import L1Checker, L1Config, L1Result
from vbw.schema import ChannelSchema, Frame, L0Result, Verdict


def frame_vector(frame: Frame, schema: ChannelSchema) -> np.ndarray:
    return np.array([frame.channels.get(c, math.nan) for c in schema.channels], dtype=float)


def l2_explanation(z: np.ndarray, channels: tuple[str, ...]) -> str:
    i = int(np.argmax(np.abs(z)))
    return (f"{channels[i]} deviates from the learned flight dynamics "
            f"({abs(float(z[i])):.1f} sigma from the predicted next value)")


class BaseDetector:
    """Streaming detector for one vehicle. Implements vbw.schema.Detector."""

    name = "base"

    def __init__(self, schema: ChannelSchema, models_dir: Path = MODELS_DIR,
                 l1_config: L1Config | None = None, l2: L2Model | None | bool = True,
                 alert: AlertConfig | None = None):
        self.schema = schema
        self.l1 = L1Checker(schema, l1_config)
        self.l2 = L2Model.load(schema, models_dir) if l2 is True else (l2 or None)
        self.alert_config = alert or AlertConfig.load(schema.name)
        self.alert = AlertState(self.alert_config)
        self.reset()

    def reset(self) -> None:
        self.l1.reset()
        self.alert.reset()
        self._ewma = np.zeros(len(self.schema.channels))
        self._last_t: float | None = None
        self._reason = ""
        self.last_l1: L1Result | None = None
        self.last_z = np.zeros(len(self.schema.channels))
        self.last_accepted = False

    # ------------------------------------------------------------------ layers

    def _l2_step(self) -> tuple[np.ndarray, float]:
        """z-scores of the newest frame and the L2 ratio (0 while the history is too short)."""
        if self.l2 is None or len(self.l1.buffer) < self.l2.needs:
            return np.zeros(len(self.schema.channels)), 0.0
        X, _ = self.l1.buffer.last(self.l2.needs)
        z = self.l2.zscore_last(X)
        self._ewma = ewma_update(self._ewma, z, self.l2.norm.ewma_alpha)
        return z, l2_ratio(z, self._ewma, self.l2.norm)

    def _explain(self, l1: L1Result, z: np.ndarray, l1_score: float, l2_score: float) -> str:
        if l1.violations and l1_score >= l2_score:
            return l1.violations[0].message
        if l2_score > 0:
            return l2_explanation(z if np.any(z) else self._ewma, self.schema.channels)
        return l1.violations[0].message if l1.violations else ""

    def _score(self, frame: Frame, l0: L0Result) -> Verdict:
        verdict = Verdict(frame.vehicle_id, frame.seq, frame.t, l0, detector=self.name)
        self.last_accepted = False
        x = frame_vector(frame, self.schema)
        if not (l0.sync_ok and l0.crc_ok) or not np.all(np.isfinite(x)):
            # Corrupted bits are L0's job; don't let them poison the physics history.
            verdict.explanation = "Link integrity failure: " + ", ".join(l0.reasons or ["non-finite values"])
            return verdict
        if self._last_t is not None and frame.t <= self._last_t:
            verdict.explanation = "Out-of-order frame skipped by physics checks"
            return verdict
        self._last_t = frame.t
        self.last_accepted = True

        l1 = self.l1.update(x, frame.t)
        z, ratio2 = self._l2_step()
        self.last_l1, self.last_z = l1, z
        verdict.l1_violations = l1.violations
        verdict.l1_score = soft_ratio(l1.max_ratio)
        verdict.l2_residuals = {c: round(float(v), 2) for c, v in zip(self.schema.channels, z)}
        verdict.l2_score = soft_ratio(ratio2)
        verdict.score = combine(verdict.l1_score, verdict.l2_score)
        if verdict.score > self.alert_config.tau:
            self._reason = self._explain(l1, z, verdict.l1_score, verdict.l2_score)
        return verdict

    # ------------------------------------------------------------------ Detector protocol

    def update(self, frame: Frame, l0: L0Result) -> Verdict:
        verdict = self._score(frame, l0)
        verdict.alert = self.alert.update(verdict.score)
        if verdict.alert:
            verdict.explanation = self._reason
        elif verdict.score > self.alert_config.tau and not verdict.explanation:
            verdict.explanation = self._reason
        return verdict

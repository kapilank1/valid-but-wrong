"""Lightweight streaming detector for the single-channel SMAP/MSL spacecraft schema (no physics).

Rolling robust z-score of the value and of its first difference against the previous `window` frames
(median / MAD), squashed to 0..1 and fed through the shared persistence / hysteresis alerting.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from vbw.detector.alerting import AlertConfig, AlertState, soft_ratio
from vbw.schema import SPACECRAFT, ChannelSchema, Frame, L0Result, Verdict

WINDOW = 250
MIN_HISTORY = 30
Z_VALUE = 6.0          # robust z that counts as ratio 1
Z_DIFF = 8.0
MAD_FLOOR = 1e-3       # SMAP/MSL values are normalized to [-1, 1]; flat channels would otherwise give MAD = 0


def robust_z(x: float, history: np.ndarray) -> float:
    med = float(np.median(history))
    mad = 1.4826 * float(np.median(np.abs(history - med)))
    return (x - med) / max(mad, MAD_FLOOR)


class SpacecraftDetector:
    name = "base"

    def __init__(self, schema: ChannelSchema = SPACECRAFT, window: int = WINDOW, alert: AlertConfig | None = None):
        self.schema = schema
        self.window = window
        self.alert_config = alert or AlertConfig(k=3, n=5, clear_frames=20, warmup=MIN_HISTORY)
        self.alert = AlertState(self.alert_config)
        self.reset()

    def reset(self) -> None:
        self._values: deque[float] = deque(maxlen=self.window)
        self._diffs: deque[float] = deque(maxlen=self.window)
        self.alert.reset()

    def update(self, frame: Frame, l0: L0Result) -> Verdict:
        verdict = Verdict(frame.vehicle_id, frame.seq, frame.t, l0, detector=self.name)
        x = float(frame.channels.get("value", np.nan))
        if not np.isfinite(x) or not (l0.sync_ok and l0.crc_ok):
            verdict.explanation = "Link integrity failure: " + ", ".join(l0.reasons or ["non-finite value"])
            return verdict
        diff = x - self._values[-1] if self._values else 0.0
        z_val = z_diff = 0.0
        if len(self._values) >= MIN_HISTORY:
            z_val = robust_z(x, np.fromiter(self._values, float))
            z_diff = robust_z(diff, np.fromiter(self._diffs, float))
        self._values.append(x)
        self._diffs.append(diff)

        ratio = max(abs(z_val) / Z_VALUE, abs(z_diff) / Z_DIFF)
        verdict.l2_residuals = {"value": round(z_val, 2), "value_diff": round(z_diff, 2)}
        verdict.l2_score = soft_ratio(ratio)
        verdict.score = verdict.l2_score
        verdict.alert = self.alert.update(verdict.score)
        if verdict.alert or verdict.score > self.alert_config.tau:
            which = "value" if abs(z_val) / Z_VALUE >= abs(z_diff) / Z_DIFF else "rate of change"
            verdict.explanation = f"Telemetry {which} is {max(abs(z_val), abs(z_diff)):.1f} robust sigma from its recent history"
        return verdict

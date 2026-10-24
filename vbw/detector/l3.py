"""HardenedDetector: BaseDetector plus an L3 discriminator P(forged) over the last L3_CONTEXT frames of features.

The discriminator (models/l3_<schema>.onnx, trained by scripts/train_l3.py on attacker-forged windows) sees the
per-frame L1 residual ratios, L2 z-scores and normalized diffs (features.py), so it can learn attack signatures
that stay under every individual threshold. Its probability is converted to a score against the clean-data
threshold in models/l3_<schema>_meta.json: l3_score = sigmoid(2·(logit − logit_threshold)), which is 0.5 exactly at
the threshold like the L1 / L2 scores. The combined score is max(noisy-OR(L1, L2), l3_score), and the threshold is
calibrated (scripts/train_l3.py) so that no clean training or validation flight alerts on L3 alone.

Without a trained model the hardened detector behaves exactly like the base one, with l3_prob = None.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from vbw.detector.alerting import AlertConfig, AlertState
from vbw.detector.base import BaseDetector
from vbw.detector.features import L3_CONTEXT, compose, diff_scale, l3_rules
from vbw.detector.l2 import MODELS_DIR
from vbw.schema import ChannelSchema, Frame, L0Result, Verdict


L3_PERSIST_K, L3_PERSIST_N = 40, 50   # L3 alerts when its score is over τ on 40 of the last 50 frames (~4 s)


def l3_paths(schema: ChannelSchema, models_dir: Path = MODELS_DIR) -> tuple[Path, Path]:
    return models_dir / f"l3_{schema.name}.onnx", models_dir / f"l3_{schema.name}_meta.json"


class L3Model:
    def __init__(self, onnx_path: Path, meta: dict):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.meta = meta
        self.logit_threshold = float(meta.get("logit_threshold", 0.0))
        self.context = int(meta.get("context", L3_CONTEXT))

    @classmethod
    def load(cls, schema: ChannelSchema, models_dir: Path = MODELS_DIR) -> "L3Model | None":
        onnx_path, meta_path = l3_paths(schema, models_dir)
        if not (onnx_path.exists() and meta_path.exists()):
            return None
        return cls(onnx_path, json.loads(meta_path.read_text()))

    def logit(self, features: np.ndarray, batch: int = 8192) -> np.ndarray:
        """[N, context, F] float32 -> logit of P(forged) [N]."""
        out = [self.session.run(None, {"x": features[i:i + batch]})[0] for i in range(0, len(features), batch)]
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def score(self, logit: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-2.0 * (logit - self.logit_threshold)))


class HardenedDetector(BaseDetector):
    name = "hardened"

    def __init__(self, schema: ChannelSchema, models_dir: Path = MODELS_DIR, l3: L3Model | None | bool = True, **kw):
        self.l3 = L3Model.load(schema, models_dir) if l3 is True else (l3 or None)
        super().__init__(schema, models_dir, **kw)

    def l3_alert_config(self) -> AlertConfig:
        """L3's own persistence: sustained (k of n frames) rather than the base 3 of 5. Forgeries the discriminator
        catches are sustained; clean transients it trips on (first-stage boostback shutdown) last 1-3 s."""
        meta = self.l3.meta if self.l3 is not None else {}
        return replace(self.alert_config, k=int(meta.get("persistence_k", L3_PERSIST_K)),
                       n=int(meta.get("persistence_n", L3_PERSIST_N)), warmup=0, meta=None)

    def reset(self) -> None:
        super().reset()
        context = self.l3.context if self.l3 is not None else L3_CONTEXT
        self._features: deque[np.ndarray] = deque(maxlen=context)
        self._prev_x: np.ndarray | None = None
        self.l3_alert = AlertState(self.l3_alert_config())

    def _frame_features(self, x: np.ndarray) -> np.ndarray:
        rules = l3_rules(self.schema)
        ratios = np.array([self.last_l1.ratios.get(r.name, 0.0) for r in rules]) if self.last_l1 else np.zeros(len(rules))
        diff = (x - self._prev_x) / diff_scale(self.schema, self.l2) if self._prev_x is not None else np.zeros_like(x)
        return compose(ratios, self.last_z, diff)

    def update(self, frame: Frame, l0: L0Result) -> Verdict:
        verdict = self._score(frame, l0)
        base_score, l3_score = verdict.score, 0.0
        if self.l3 is not None and self.last_accepted:
            x = self.l1.buffer.last(1)[0][0]
            self._features.append(self._frame_features(x))
            self._prev_x = x.copy()
            if len(self._features) == self._features.maxlen:
                logit = float(self.l3.logit(np.stack(self._features)[None])[0])
                p = 1.0 / (1.0 + np.exp(-logit))
                verdict.l3_prob = round(p, 4)
                l3_score = float(self.l3.score(np.array([logit]))[0])
                # Its own stream, not part of the noisy-OR: L1+L2 already sit near τ on some clean OCR glitches, so
                # adding even a small L3 score there would alert. L3's threshold is calibrated on its own instead.
                verdict.score = max(base_score, l3_score)
        base_alert = self.alert.update(base_score)
        l3_alert = self.l3_alert.update(l3_score)
        if l3_alert and not base_alert:
            prob = f"P(forged) = {verdict.l3_prob:.2f}, " if verdict.l3_prob is not None else ""
            self._reason = f"Telemetry pattern matches forged-link signatures (discriminator {prob}sustained)"
        verdict.alert = base_alert or l3_alert
        if verdict.alert or (verdict.score > self.alert_config.tau and not verdict.explanation):
            verdict.explanation = self._reason
        return verdict

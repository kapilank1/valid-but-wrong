"""HardenedDetector: BaseDetector plus an L3 discriminator P(forged) over the last L3_CONTEXT frames of features.

The discriminator (models/l3_<schema>.onnx, trained by scripts/train_l3.py on attacker-forged windows) sees the
per-frame L1 residual ratios, L2 z-scores and normalized diffs (features.py), so it can learn attack signatures
that stay under every individual threshold. Its probability is converted to a ratio against the clean-data
threshold in models/l3_<schema>_meta.json: l3_score = sigmoid(2·(logit − logit_threshold)), which is 0.5 exactly at
the threshold like the L1 / L2 scores, and is folded into the same noisy-OR.

Without a trained model the hardened detector behaves exactly like the base one, with l3_prob = None.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from vbw.detector.alerting import combine
from vbw.detector.base import BaseDetector
from vbw.detector.features import L3_CONTEXT, compose, diff_scale
from vbw.detector.l2 import MODELS_DIR
from vbw.schema import ChannelSchema, Frame, L0Result, Verdict


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

    def reset(self) -> None:
        super().reset()
        context = self.l3.context if self.l3 is not None else L3_CONTEXT
        self._features: deque[np.ndarray] = deque(maxlen=context)
        self._prev_x: np.ndarray | None = None

    def _frame_features(self, x: np.ndarray) -> np.ndarray:
        rules = self.l1.rules
        ratios = np.array([self.last_l1.ratios.get(r.name, 0.0) for r in rules]) if self.last_l1 else np.zeros(len(rules))
        diff = (x - self._prev_x) / diff_scale(self.schema, self.l2) if self._prev_x is not None else np.zeros_like(x)
        return compose(ratios, self.last_z, diff)

    def update(self, frame: Frame, l0: L0Result) -> Verdict:
        verdict = self._score(frame, l0)
        if self.l3 is not None and self.last_accepted:
            x = self.l1.buffer.last(1)[0][0]
            self._features.append(self._frame_features(x))
            self._prev_x = x.copy()
            if len(self._features) == self._features.maxlen:
                logit = float(self.l3.logit(np.stack(self._features)[None])[0])
                p = 1.0 / (1.0 + np.exp(-logit))
                verdict.l3_prob = round(p, 4)
                l3_score = float(self.l3.score(np.array([logit]))[0])
                verdict.score = combine(verdict.l1_score, verdict.l2_score, l3_score)
                if l3_score > max(verdict.l1_score, verdict.l2_score) and verdict.score > self.alert_config.tau:
                    self._reason = (f"Telemetry pattern matches forged-link signatures "
                                    f"(discriminator P(forged) = {p:.2f})")
        verdict.alert = self.alert.update(verdict.score)
        if verdict.alert or (verdict.score > self.alert_config.tau and not verdict.explanation):
            verdict.explanation = self._reason
        return verdict

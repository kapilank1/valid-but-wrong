"""The hardened detector's L3 is its own alert stream: it needs a sustained forgery verdict, and never mixes into L1+L2."""

from __future__ import annotations

import numpy as np

from tests.detector_synth import falcon9_track
from vbw.detector.l3 import HardenedDetector
from vbw.detector.physics import FALCON9_RULES, L1Config
from vbw.schema import FALCON9, Frame, L0Result

L1_DEFAULTS = L1Config("falcon9", {r.name: r.default_threshold for r in FALCON9_RULES}, {})


class _FixedL3:
    """Stands in for L3Model: a constant logit, threshold 0, sustained persistence 40 of 50."""

    context = 40
    meta = {"persistence_k": 40, "persistence_n": 50}
    logit_threshold = 0.0

    def __init__(self, logit: float):
        self._logit = logit

    def logit(self, features):
        return np.full(len(features), self._logit)

    def score(self, logit):
        return 1.0 / (1.0 + np.exp(-2.0 * (np.asarray(logit) - self.logit_threshold)))


def _run(logit: float, frames: int = 150) -> list:
    t, X = falcon9_track(T=frames, noise=1.0)
    det = HardenedDetector(FALCON9, l3=_FixedL3(logit), l1_config=L1_DEFAULTS, l2=False)
    return [det.update(Frame("f9", FALCON9.schema_id, i, float(t[i]),
                             {c: float(X[i, j]) for j, c in enumerate(FALCON9.channels)}), L0Result())
            for i in range(frames)]


def test_sustained_l3_verdict_alerts_on_its_own_after_k_frames():
    verdicts = _run(logit=5.0)
    first = next(i for i, v in enumerate(verdicts) if v.alert)
    assert first == 39 + 39                    # 40 frames to fill the context, then 40 frames over τ
    assert "forged-link signatures" in verdicts[first].explanation
    assert all(v.l1_score < 0.5 and v.l2_score < 0.5 for v in verdicts)   # base layers never asked for it


def test_confident_clean_l3_never_alerts():
    assert not any(v.alert for v in _run(logit=-5.0))

"""Published Telemanom numbers, transcribed from the paper (verified against the arXiv PDF).

Source: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
Thresholding", KDD 2018, arXiv:1802.04431v3, Table 2, row "Non-Parametric w/ Pruning (p = 0.13)".
The same precision/recall/F0.5 appear in the khundman/telemanom README.

The paper reports precision, recall and F0.5 only. F1 here is computed from the reported precision and
recall (marked `f1_derived`). The printed F0.5 values do not follow from the printed precision/recall
(e.g. SMAP P = R = 85.5% implies F0.5 = 0.855, printed 0.71), so compare on precision/recall/F1.
"""

from __future__ import annotations


def _f1(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4)


_REPORTED = {
    "smap": {"precision": 0.855, "recall": 0.855, "f0_5_reported": 0.71, "anomalies": 69, "channels": 55},
    "msl": {"precision": 0.926, "recall": 0.694, "f0_5_reported": 0.69, "anomalies": 36, "channels": 27},
    "combined": {"precision": 0.875, "recall": 0.800, "f0_5_reported": 0.71, "anomalies": 105, "channels": 82},
}

BASELINE: dict = {
    "name": "Telemanom (Hundman et al., KDD 2018)",
    "source_url": "https://arxiv.org/abs/1802.04431",
    "source_detail": "Table 2, Non-Parametric w/ Pruning (p = 0.13); also khundman/telemanom README",
    "f1_derived": True,
    **{k: {**v, "f1": _f1(v["precision"], v["recall"])} for k, v in _REPORTED.items()},
}

"""Scoring for SMAP/MSL detections.

Event-level (the paper's protocol, Hundman et al. 2018 Sec. 4.1):
  * a labeled sequence is a true positive if any predicted sequence overlaps it (counted once);
  * a labeled sequence no predicted sequence overlaps is a false negative;
  * a predicted sequence that overlaps no labeled sequence is a false positive.

Point-adjusted (Xu et al. 2018 style), reported only as an optimistic reference: point-wise scoring
where detecting any point of a labeled segment credits the whole segment. It is known to inflate scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

Sequence = tuple[int, int]


@dataclass
class EventCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "EventCounts") -> "EventCounts":
        return EventCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def to_dict(self) -> dict:
        return asdict(self)


def overlaps(a: Sequence, b: Sequence) -> bool:
    """Inclusive ranges share at least one index."""
    return a[0] <= b[1] and b[0] <= a[1]


def match_events(predicted: list[Sequence], labeled: list[Sequence]) -> tuple[EventCounts, list[bool], list[bool]]:
    """Event-level counts plus which labeled sequences were hit and which predictions were false alarms."""
    hit = [any(overlaps(p, lab) for p in predicted) for lab in labeled]
    false_alarm = [not any(overlaps(p, lab) for lab in labeled) for p in predicted]
    counts = EventCounts(tp=sum(hit), fp=sum(false_alarm), fn=len(labeled) - sum(hit))
    return counts, hit, false_alarm


def precision_recall_f(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall, F1 and F0.5 (the paper's headline metric). Zero when undefined."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(beta: float) -> float:
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {"precision": precision, "recall": recall, "f1": f_beta(1.0), "f0_5": f_beta(0.5)}


def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Mark every point of a labeled segment as detected if any of its points was predicted."""
    adjusted = pred.astype(bool).copy()
    lab = label.astype(bool)
    padded = np.concatenate(([False], lab, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for s, e in zip(edges[::2], edges[1::2]):
        if adjusted[s:e].any():
            adjusted[s:e] = True
    return adjusted


def point_adjusted_counts(pred: np.ndarray, label: np.ndarray) -> EventCounts:
    """Point-wise TP/FP/FN after point adjustment (counts are points, not events)."""
    adj = point_adjust(pred, label)
    lab = label.astype(bool)
    return EventCounts(tp=int((adj & lab).sum()), fp=int((adj & ~lab).sum()), fn=int((~adj & lab).sum()))

"""Turn per-channel predictions into detections and event-level scores per spacecraft."""

from __future__ import annotations

import numpy as np

from vbw.bench.data import LabelRow, sequences_to_mask
from vbw.bench.metrics import EventCounts, match_events, point_adjusted_counts, precision_recall_f
from vbw.bench.model import ChannelPrediction
from vbw.bench.threshold import ThresholdConfig, detect


def evaluate_row(row: LabelRow, pred: ChannelPrediction, cfg: ThresholdConfig) -> dict:
    """Detect anomalies in one labeled stream and score them against its labels."""
    errors = np.abs(pred.y_hat - pred.y_true)
    det = detect(errors, pred.y_true, cfg)
    labeled = list(row.sequences)
    counts, hit, false_alarm = match_events(det.sequences, labeled)
    pred_mask = sequences_to_mask(det.sequences, row.num_values)
    label_mask = sequences_to_mask(labeled, row.num_values)
    pa = point_adjusted_counts(pred_mask, label_mask)
    span = float(np.ptp(pred.y_true)) or 1.0
    return {
        "row": row.row,
        "chan_id": row.chan_id,
        "spacecraft": row.spacecraft,
        "num_values": row.num_values,
        "labeled_sequences": [list(s) for s in labeled],
        "classes": list(row.classes),
        "predicted_sequences": [list(s) for s in det.sequences],
        "scores": [round(s, 4) for s in det.scores],
        "hit": hit,
        "false_alarm": false_alarm,
        **counts.to_dict(),
        "pa": pa.to_dict(),
        "normalized_error": float(errors.mean() / span),
        "epochs_run": pred.epochs_run,
        "best_val_loss": pred.best_val_loss,
        "n_train_windows": pred.n_train_windows,
        "train_s": round(pred.train_s, 2),
        "device": pred.device,
    }


def _class_recall(rows: list[dict]) -> dict[str, dict]:
    by_class: dict[str, list[bool]] = {}
    for r in rows:
        for cls, h in zip(r["classes"], r["hit"]):
            by_class.setdefault(cls, []).append(h)
    return {c: {"recall": sum(h) / len(h), "hit": sum(h), "total": len(h)} for c, h in sorted(by_class.items())}


def summarize(rows: list[dict]) -> dict:
    """Aggregate event-level and point-adjusted counts over a set of evaluated rows."""
    ev = sum((EventCounts(r["tp"], r["fp"], r["fn"]) for r in rows), EventCounts())
    pa = sum((EventCounts(**r["pa"]) for r in rows), EventCounts())
    pa_scores = precision_recall_f(pa.tp, pa.fp, pa.fn)
    return {
        **{k: round(v, 4) for k, v in precision_recall_f(ev.tp, ev.fp, ev.fn).items()},
        **ev.to_dict(),
        "channels": len(rows),
        "unique_channels": len({r["chan_id"] for r in rows}),
        "anomalies": ev.tp + ev.fn,
        "recall_by_class": _class_recall(rows),
        "point_adjusted_optimistic": {
            "precision": round(pa_scores["precision"], 4),
            "recall": round(pa_scores["recall"], 4),
            "f1": round(pa_scores["f1"], 4),
            "note": "point-wise after point adjustment; inflates scores, not comparable to event-level",
        },
        "mean_normalized_error": round(float(np.mean([r["normalized_error"] for r in rows])), 4) if rows else None,
    }


def summarize_by_spacecraft(rows: list[dict]) -> dict[str, dict]:
    return {
        "smap": summarize([r for r in rows if r["spacecraft"] == "SMAP"]),
        "msl": summarize([r for r in rows if r["spacecraft"] == "MSL"]),
        "combined": summarize(rows),
    }

"""SMAP/MSL loaders and label parsing (Hundman et al., KDD 2018 / telemanom layout).

Each row of labeled_anomalies.csv is one evaluated stream. Telemanom evaluates the CSV row by row, so the
duplicated P-2 row is kept (55 SMAP rows / 69 anomalies, 27 MSL rows / 36 anomalies, as in the paper).
Channel files that have no CSV row (T-10 in the TranAD mirror) are not evaluated.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "smap_msl"

Sequence = tuple[int, int]  # inclusive [start, end] index range into the test array


@dataclass(frozen=True)
class LabelRow:
    """One row of labeled_anomalies.csv."""

    row: int                     # 0-based row index in the CSV (disambiguates the duplicated P-2)
    chan_id: str
    spacecraft: str              # "SMAP" | "MSL"
    sequences: tuple[Sequence, ...]
    classes: tuple[str, ...]
    num_values: int


def parse_sequences(text: str) -> tuple[Sequence, ...]:
    """Parse "[[s, e], ...]" into inclusive (start, end) tuples. Uses JSON, never eval."""
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError(f"anomaly_sequences must be a list, got {text!r}")
    out: list[Sequence] = []
    for pair in raw:
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, int) for v in pair)):
            raise ValueError(f"bad sequence {pair!r} in {text!r}")
        start, end = pair
        if start < 0 or end < start:
            raise ValueError(f"bad sequence bounds {pair!r}")
        out.append((start, end))
    return tuple(out)


def parse_classes(text: str) -> tuple[str, ...]:
    """Parse "[point, contextual]" (unquoted, so not JSON) into a tuple of class names."""
    inner = text.strip().removeprefix("[").removesuffix("]")
    return tuple(part.strip() for part in inner.split(",") if part.strip())


def load_labels(path: Path | None = None) -> list[LabelRow]:
    """Read every row of labeled_anomalies.csv, in file order."""
    path = path or DATA_DIR / "labeled_anomalies.csv"
    rows: list[LabelRow] = []
    with open(path, newline="") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            rows.append(
                LabelRow(
                    row=i,
                    chan_id=rec["chan_id"].strip(),
                    spacecraft=rec["spacecraft"].strip(),
                    sequences=parse_sequences(rec["anomaly_sequences"]),
                    classes=parse_classes(rec["class"]),
                    num_values=int(rec["num_values"]),
                )
            )
    return rows


def load_array(path: Path) -> np.ndarray:
    """Load one untrusted .npy file as a finite float32 [timesteps, features] array."""
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"{path}: expected 2-D array, got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: contains non-finite values")
    return arr


def load_channel(chan_id: str, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) arrays for a channel. Column 0 is telemetry, the rest are command one-hots."""
    data_dir = data_dir or DATA_DIR
    if "/" in chan_id or "\\" in chan_id or chan_id.startswith("."):
        raise ValueError(f"bad channel id {chan_id!r}")
    train = load_array(data_dir / "train" / f"{chan_id}.npy")
    test = load_array(data_dir / "test" / f"{chan_id}.npy")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{chan_id}: train/test feature mismatch {train.shape} vs {test.shape}")
    return train, test


def unique_channels(rows: list[LabelRow]) -> list[str]:
    """Channel ids in first-appearance order (P-2 is listed twice but trained once)."""
    seen: dict[str, None] = {}
    for r in rows:
        seen.setdefault(r.chan_id, None)
    return list(seen)


def sequences_to_mask(sequences: tuple[Sequence, ...] | list[Sequence], length: int) -> np.ndarray:
    """Boolean mask of length `length` that is True inside any inclusive sequence (clipped to bounds)."""
    mask = np.zeros(length, dtype=bool)
    for start, end in sequences:
        mask[max(start, 0) : min(end, length - 1) + 1] = True
    return mask

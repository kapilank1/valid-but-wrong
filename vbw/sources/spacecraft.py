"""Replay of NASA SMAP / MSL spacecraft telemetry channels (Hundman et al., KDD 2018) as SPACECRAFT frames.

Each <chan>.npy holds one normalized telemetry value (column 0) plus command one-hot columns; only column 0 goes on
the link, as channel "value". The data has no timestamps, so mission time is t = index / rate_hz.

Labels come from labeled_anomalies.csv and index the *test* split ([start, end], inclusive). Two quirks of that file
are handled here: P-2 is listed twice (the first row wins), and T-10 has no row at all (it replays with no labels;
its spacecraft is inferred from the column count: SMAP files have 25 columns, MSL files 55).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from vbw.schema import SPACECRAFT, ChannelSchema, Frame
from vbw.sources.base import data_dir, seq_at

Dataset = Literal["smap", "msl"]
Split = Literal["train", "test"]
COLUMNS_BY_DATASET: dict[str, int] = {"smap": 25, "msl": 55}


def smap_msl_dir() -> Path:
    return data_dir() / "raw" / "smap_msl"


@dataclass(frozen=True)
class ChannelLabels:
    chan_id: str
    dataset: str                          # "smap" | "msl"
    anomaly_sequences: tuple[tuple[int, int], ...]
    classes: tuple[str, ...]
    num_values: int


def _parse_classes(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.strip("[]").split(",") if part.strip())


@lru_cache(maxsize=1)
def channel_labels() -> dict[str, ChannelLabels]:
    """chan_id -> labels, first occurrence wins for duplicated rows."""
    table = pd.read_csv(smap_msl_dir() / "labeled_anomalies.csv")
    labels: dict[str, ChannelLabels] = {}
    for row in table.to_dict("records"):
        chan_id = row["chan_id"]
        if chan_id in labels:
            continue
        sequences = [(int(a), int(b)) for a, b in json.loads(row["anomaly_sequences"])]
        classes = _parse_classes(row["class"])
        ordered = sorted(zip(sequences, classes))  # the CSV doesn't always list anomalies chronologically
        labels[chan_id] = ChannelLabels(
            chan_id, row["spacecraft"].lower(), tuple(s for s, _ in ordered), tuple(c for _, c in ordered),
            int(row["num_values"]),
        )
    return labels


def load_channel(chan_id: str, split: Split = "test") -> np.ndarray:
    """Raw [T, 1 + n_commands] array. Downloaded data is untrusted: never unpickle it."""
    return np.load(smap_msl_dir() / split / f"{chan_id}.npy", allow_pickle=False)


def infer_dataset(array: np.ndarray) -> str | None:
    for dataset, columns in COLUMNS_BY_DATASET.items():
        if array.shape[1] == columns:
            return dataset
    return None


@dataclass
class SpacecraftSource:
    """Source protocol implementation for one SMAP/MSL channel. Deterministic: replays the recorded values."""

    dataset: Dataset
    chan_id: str
    split: Split = "test"
    rate_hz: float = 10.0
    seq_start: int = 0
    schema: ChannelSchema = field(default=SPACECRAFT, init=False)
    vehicle_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.dataset not in COLUMNS_BY_DATASET:
            raise ValueError(f"dataset must be 'smap' or 'msl', got {self.dataset!r}")
        raw = load_channel(self.chan_id, self.split)
        labels = channel_labels().get(self.chan_id)
        actual = labels.dataset if labels else infer_dataset(raw)
        if actual != self.dataset:
            raise ValueError(f"channel {self.chan_id} belongs to {actual}, not {self.dataset}")
        self.values = raw[:, 0].astype(float)
        self.labels = labels
        self.vehicle_id = f"{self.dataset}-{self.chan_id}"

    @property
    def duration_s(self) -> float:
        return len(self.values) / self.rate_hz

    @property
    def anomaly_sequences(self) -> list[list[int]]:
        """[start, end] sample-index pairs (inclusive). Labels only exist for the test split."""
        if self.split != "test" or self.labels is None:
            return []
        return [[a, b] for a, b in self.labels.anomaly_sequences]

    @property
    def anomaly_t_ranges(self) -> list[list[float]]:
        return [[a / self.rate_hz, b / self.rate_hz] for a, b in self.anomaly_sequences]

    @property
    def events(self) -> dict[str, float]:
        return {f"anomaly_{i + 1}": t0 for i, (t0, _) in enumerate(self.anomaly_t_ranges)}

    def frames(self) -> Iterator[Frame]:
        for i, value in enumerate(self.values):
            yield Frame(self.vehicle_id, SPACECRAFT.schema_id, seq_at(i, self.seq_start), i / self.rate_hz,
                        {"value": float(value)})

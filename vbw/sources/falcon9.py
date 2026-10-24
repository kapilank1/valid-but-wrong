"""Replay of real Falcon 9 ascent telemetry (webcast-derived, 10 Hz) as FALCON9 frames.

The processed CSVs are the ground truth. Frames add small seeded white noise so a replay looks like a sensor
feed rather than a spreadsheet, while staying well inside the data's own inconsistency (on thaicom-8 the noise
moves the 1-s |d(alt)/dt - vz| median from 1.30 to 1.50 m/s and its p95 from 16.5 to 16.6 m/s):

    channel     sigma       why this size
    speed       0.1 m/s     INS velocity-grade; the data's own |speed - hypot(vz, vx)| median is 0.14-0.35 m/s
    altitude    0.5 m       gives ~0.7 m/s of noise on a 1-s finite difference of altitude
    vz, vx      0.1 m/s     same grade as speed
    downrange   1.0 m       range is ~1000x altitude-rate scale, so a looser absolute sigma
    accel       0.02 m/s^2  accelerometer-grade
    angle       0.02 deg    flight-path angle from the INS
    q           0.2 %       multiplicative, so dynamic pressure stays >= 0 and reads exactly 0 in vacuum
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from vbw.schema import FALCON9, ChannelSchema, Frame
from vbw.sources.base import data_dir, resolve_seed, seq_at

RATE_HZ = 10.0

NOISE_SIGMA: dict[str, float] = {
    "speed": 0.1,
    "altitude": 0.5,
    "vz": 0.1,
    "vx": 0.1,
    "downrange": 1.0,
    "accel": 0.02,
    "angle": 0.02,
}
Q_RELATIVE_SIGMA = 0.002


def falcon9_dir() -> Path:
    return data_dir() / "processed" / "falcon9"


@lru_cache(maxsize=1)
def falcon9_index() -> tuple[dict, ...]:
    """Launch metadata from index.json (slug, mission_name, stage, duration_s, events)."""
    return tuple(json.loads((falcon9_dir() / "index.json").read_text()))


def launch_info(slug: str) -> dict:
    for launch in falcon9_index():
        if launch["slug"] == slug:
            return launch
    raise KeyError(f"unknown Falcon 9 launch {slug!r}")


def add_sensor_noise(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy copy of a [T, C] array in FALCON9 channel order."""
    noisy = truth.copy()
    for name, sigma in NOISE_SIGMA.items():
        i = FALCON9.index(name)
        noisy[:, i] += rng.normal(0.0, sigma, size=len(noisy))
    iq = FALCON9.index("q")
    noisy[:, iq] *= 1.0 + rng.normal(0.0, Q_RELATIVE_SIGMA, size=len(noisy))
    ispeed = FALCON9.index("speed")
    noisy[:, ispeed] = np.maximum(noisy[:, ispeed], 0.0)  # a speed magnitude can't read negative on the pad
    return noisy


@dataclass
class Falcon9Source:
    """Source protocol implementation for one launch. frames() is repeatable: same seed, same noise."""

    slug: str
    seed: int | None = None
    noise: bool = True
    seq_start: int = 0
    schema: ChannelSchema = field(default=FALCON9, init=False)
    rate_hz: float = field(default=RATE_HZ, init=False)
    vehicle_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.info = launch_info(self.slug)
        self.seed = resolve_seed(self.seed)
        self.vehicle_id = f"falcon9-{self.slug}"
        self._truth = pd.read_csv(falcon9_dir() / f"{self.slug}.csv")

    @property
    def mission_name(self) -> str:
        return self.info["mission_name"]

    @property
    def duration_s(self) -> float:
        return float(self.info["duration_s"])

    @property
    def events(self) -> dict[str, float]:
        return {name: float(t) for name, t in self.info.get("events", {}).items()}

    def truth_dataframe(self) -> pd.DataFrame:
        """Noise-free CSV values: columns t + FALCON9 channels, SI units."""
        return self._truth.copy()

    def frames(self) -> Iterator[Frame]:
        t = self._truth["t"].to_numpy(float)
        values = self._truth[list(FALCON9.channels)].to_numpy(float)
        if self.noise:
            values = add_sensor_noise(values, np.random.default_rng(self.seed))
        for i, (ti, row) in enumerate(zip(t, values)):
            channels = dict(zip(FALCON9.channels, map(float, row)))
            yield Frame(self.vehicle_id, FALCON9.schema_id, seq_at(i, self.seq_start), float(ti), channels)

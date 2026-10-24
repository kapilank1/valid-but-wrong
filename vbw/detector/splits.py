"""Train / holdout split of the real Falcon 9 launches, shared by every detector script and the attacker.

Holdout = every 5th launch in index order (a mix of GTO, LEO and polar missions). Calibration and training never
see these, so the false-alert numbers measured on them are honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vbw.schema import FALCON9

ROOT = Path(__file__).resolve().parents[2]
FALCON9_DIR = ROOT / "data" / "processed" / "falcon9"
HOLDOUT_EVERY = 5
HOLDOUT_OFFSET = 2


def falcon9_index() -> list[dict]:
    return json.loads((FALCON9_DIR / "index.json").read_text())


def all_slugs() -> list[str]:
    return [launch["slug"] for launch in falcon9_index()]


def holdout_slugs() -> list[str]:
    return [s for i, s in enumerate(all_slugs()) if i % HOLDOUT_EVERY == HOLDOUT_OFFSET]


def train_slugs() -> list[str]:
    held = set(holdout_slugs())
    return [s for s in all_slugs() if s not in held]


def load_falcon9(slug: str) -> tuple[np.ndarray, np.ndarray]:
    """Clean launch as (t [T], X [T, C]) in FALCON9 channel order."""
    df = pd.read_csv(FALCON9_DIR / f"{slug}.csv")
    return df["t"].to_numpy(float), df[list(FALCON9.channels)].to_numpy(float)


def falcon9_events(slug: str) -> dict[str, float]:
    for launch in falcon9_index():
        if launch["slug"] == slug:
            return {k: float(v) for k, v in launch.get("events", {}).items()}
    return {}

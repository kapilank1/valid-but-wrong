"""Clean training / validation / test runs for calibrating and training the detector.

A run is what the ground station would decode for one scenario: sensor noise from the source's own noise model,
then float32 rounding from the wire format. Splits:

    train  = splits.train_slugs()            L1 thresholds, L2 / L3 training
    val    = holdout_slugs()[0::2]           L2 early stopping, alert threshold τ
    test   = holdout_slugs()[1::2]           reported numbers only
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from vbw.detector import splits
from vbw.schema import QUAD, ChannelSchema

SPLITS = ("train", "val", "test", "holdout")


@dataclass
class Run:
    scenario: str
    t: np.ndarray        # [T] mission time, s
    X: np.ndarray        # [T, C] reported values (noisy, float32-rounded), schema channel order
    truth: np.ndarray    # [T, C] noise-free values


def wire_round(X: np.ndarray) -> np.ndarray:
    """Values as they come off the link: the codec carries float32."""
    return X.astype(np.float32).astype(np.float64)


def _split_items(items: list[str], held: list[str], split: str) -> list[str]:
    if split == "train":
        return [s for s in items if s not in set(held)]
    if split == "val":
        return held[0::2]
    if split == "test":
        return held[1::2]
    if split == "holdout":
        return list(held)
    raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def falcon9_runs(split: str, seeds: tuple[int, ...] = (0,), noise: bool = True) -> list[Run]:
    from vbw.sources.falcon9 import add_sensor_noise

    runs = []
    for slug in _split_items(splits.all_slugs(), splits.holdout_slugs(), split):
        t, truth = splits.load_falcon9(slug)
        for seed in seeds:
            # Seed per (launch, seed) so different launches never share a noise sequence.
            rng = np.random.default_rng([seed, zlib.crc32(slug.encode())])
            X = add_sensor_noise(truth, rng) if noise else truth.copy()
            runs.append(Run(f"falcon9:{slug}", t, wire_round(X), truth))
    return runs


def quad_scenarios() -> list[str]:
    """Quad scenario ids from the sources registry (empty if the quad sim isn't built yet)."""
    try:
        from vbw.sources.registry import list_scenarios
    except ImportError:
        return []
    return [s["id"] for s in list_scenarios() if s.get("schema") == "quad"]


def quad_runs(split: str, seeds: tuple[int, ...] = (0,)) -> list[Run]:
    from vbw.sources.registry import make_source

    ids = quad_scenarios()
    held = [s for i, s in enumerate(ids) if i % splits.HOLDOUT_EVERY == splits.HOLDOUT_OFFSET % max(len(ids), 1)]
    runs = []
    for scenario in _split_items(ids, held, split):
        for seed in seeds:
            frames = list(make_source(scenario, seed=seed).frames())
            t = np.array([f.t for f in frames])
            X = np.array([[f.channels[c] for c in QUAD.channels] for f in frames])
            runs.append(Run(scenario, t, wire_round(X), X.copy()))
    return runs


def clean_runs(schema: ChannelSchema, split: str, seeds: tuple[int, ...] = (0,)) -> list[Run]:
    if schema.name == "falcon9":
        return falcon9_runs(split, seeds)
    if schema.name == "quad":
        return quad_runs(split, seeds)
    raise ValueError(f"no clean runs for schema {schema.name}")

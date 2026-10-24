"""Small helpers shared by every telemetry source: data location, seeding, sequence counters."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SEQ_MODULO = 1 << 16  # Frame.seq is a uint16 on the wire


def data_dir() -> Path:
    """Repo data/ directory, overridable with VBW_DATA_DIR (e.g. inside a container)."""
    override = os.environ.get("VBW_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data"


def resolve_seed(seed: int | None) -> int:
    """Turn an optional seed into a concrete one, so a source replays identically every time frames() is called."""
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().generate_state(1)[0])


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Independent generators derived from one seed (e.g. process noise vs sensor noise)."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def seq_at(index: int, start: int = 0) -> int:
    """Sequence number of the index-th frame of a stream, wrapping like the uint16 on the wire."""
    return (start + index) % SEQ_MODULO

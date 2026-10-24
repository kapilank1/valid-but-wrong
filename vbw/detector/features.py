"""Forged-window dataset I/O and the per-frame features the L3 discriminator sees.

Shared by the attacker (which writes data/cache/forged_*.npz) and the hardened detector (which trains on them and
computes the same features while streaming). Format: see vbw/detector/README.md.

Per-frame feature vector (F = R + 2C for R rules and C channels):
    L1 signed residual / threshold per rule        clipped to ±10
    L2 signed robust z per channel / 5             clipped to ±10
    first difference per channel / L2 delta std    clipped to ±10
The discriminator looks at the last L3_CONTEXT frames; the earlier frames of each window are history for the
10 s L1 integration windows and the L2 GRU, so offline features equal the streaming ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vbw.detector.l2 import L2Model
from vbw.detector.physics import RULES_BY_SCHEMA, L1Config, max_lookback
from vbw.schema import SCHEMAS_BY_NAME, ChannelSchema

L3_CONTEXT = 40
L3_WINDOWS = {"falcon9": 160, "quad": 160}
FEATURE_CLIP = 10.0
Z_SCALE = 5.0


def l3_window(schema: ChannelSchema) -> int:
    w = L3_WINDOWS.get(schema.name, 160)
    needed = L3_CONTEXT + max_lookback(schema) + 1
    assert w >= needed, f"L3 window {w} too short for {schema.name} (needs {needed})"
    return w


# --------------------------------------------------------------------------- npz I/O


@dataclass
class ForgedSet:
    X: np.ndarray          # float32 [N, W, C]
    t: np.ndarray          # float64 [N, W]
    y: np.ndarray          # int8 [N]
    attack: np.ndarray     # str [N]
    scenario: np.ndarray   # str [N]
    schema: str
    window: int

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "ForgedSet":
        return ForgedSet(self.X[mask], self.t[mask], self.y[mask], self.attack[mask], self.scenario[mask],
                         self.schema, self.window)

    @staticmethod
    def concat(sets: list["ForgedSet"]) -> "ForgedSet":
        first = sets[0]
        return ForgedSet(*(np.concatenate([getattr(s, k) for s in sets]) for k in ("X", "t", "y", "attack", "scenario")),
                         first.schema, first.window)


def _validate(X, t, y, attack, scenario, schema: str, window: int) -> None:
    n_ch = len(SCHEMAS_BY_NAME[schema].channels)
    if X.ndim != 3 or X.shape[1] != window or X.shape[2] != n_ch:
        raise ValueError(f"X must be [N, {window}, {n_ch}], got {X.shape}")
    n = X.shape[0]
    if t.shape != (n, window) or y.shape != (n,) or attack.shape != (n,) or scenario.shape != (n,):
        raise ValueError("t / y / attack / scenario shapes don't match X")


def save_forged_npz(path: Path, X, t, y, attack, scenario, schema: str, window: int) -> None:
    X, t, y = np.asarray(X, np.float32), np.asarray(t, np.float64), np.asarray(y, np.int8)
    attack, scenario = np.asarray(attack).astype(str), np.asarray(scenario).astype(str)
    _validate(X, t, y, attack, scenario, schema, int(window))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, t=t, y=y, attack=attack, scenario=scenario,
                        schema=np.array(schema), window=np.array(int(window)))


def load_forged_npz(path: Path) -> ForgedSet:
    with np.load(path, allow_pickle=False) as d:
        s = ForgedSet(d["X"].astype(np.float32), d["t"].astype(np.float64), d["y"].astype(np.int8),
                      d["attack"].astype(str), d["scenario"].astype(str), str(d["schema"]), int(d["window"]))
    _validate(s.X, s.t, s.y, s.attack, s.scenario, s.schema, s.window)
    return s


# --------------------------------------------------------------------------- features


def feature_names(schema: ChannelSchema) -> list[str]:
    rules = [f"l1:{r.name}" for r in RULES_BY_SCHEMA.get(schema.name, ())]
    return rules + [f"l2z:{c}" for c in schema.channels] + [f"diff:{c}" for c in schema.channels]


def compose(ratios: np.ndarray, z: np.ndarray, diff_norm: np.ndarray) -> np.ndarray:
    """One frame (or [..., ] frames) of features from its three parts. Streaming and offline both call this."""
    parts = [np.nan_to_num(ratios), np.nan_to_num(z) / Z_SCALE, np.nan_to_num(diff_norm)]
    return np.clip(np.concatenate(parts, axis=-1), -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)


def diff_scale(schema: ChannelSchema, l2: L2Model | None) -> np.ndarray:
    return np.asarray(l2.norm.dstd) if l2 is not None else np.ones(len(schema.channels))


def l3_features(X: np.ndarray, t: np.ndarray, schema: ChannelSchema, l1_config: L1Config | None = None,
                l2: L2Model | None = None, context: int = L3_CONTEXT, chunk: int = 2048) -> np.ndarray:
    """[N, W, C] reported windows (+ [N, W] times) -> float32 [N, context, F] features of their last frames."""
    l1_config = l1_config or L1Config.load(schema)
    rules = RULES_BY_SCHEMA.get(schema.name, ())
    dscale = diff_scale(schema, l2)
    out = []
    for i in range(0, len(X), chunk):
        Xc, tc = np.asarray(X[i:i + chunk], np.float64), np.asarray(t[i:i + chunk], np.float64)
        ratios = np.stack([r.residual(Xc, tc, l1_config.params)[:, -context:] / l1_config.thresholds[r.name]
                           for r in rules], -1) if rules else np.zeros(Xc.shape[:1] + (context, 0))
        z = l2.zscore_series(Xc, last=context)[:, -context:] if l2 is not None else np.zeros((len(Xc), context, Xc.shape[-1]))
        diff = (np.diff(Xc, axis=1)[:, -context:]) / dscale
        out.append(compose(ratios, z, diff))
    return np.concatenate(out) if out else np.zeros((0, context, len(feature_names(schema))), np.float32)

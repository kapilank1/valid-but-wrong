"""Nonparametric dynamic error thresholding (Hundman et al., KDD 2018, Sec. 3.2-3.3).

Follows the reference implementation (khundman/telemanom, errors.py):
  1. EWMA-smooth absolute one-step prediction errors (span = batch_size * window_size * smoothing_perc).
  2. Walk the errors in batches of `batch_size`, each judged against a trailing window of
     h = batch_size * window_size smoothed errors.
  3. Per window, pick epsilon = mean + z*std over a z grid, maximising
     (dmean/mean + dstd/std) / (|E_seq|^2 + |e_a|), with guardrails (<= 5 sequences, < 50% flagged).
     The same is done on errors mirrored about their mean, so sudden drops in error also count.
  4. Pad flagged points by `error_buffer`, then prune: sort sequence maxima descending, append the
     largest non-anomalous error, and reclassify as nominal every sequence after the last step whose
     percent decrease reaches `p`.

Deliberate differences from the reference code (none use test labels):
  * no per-channel special case (telemanom skips the start-of-series smoothing fix for C-2 only);
  * the final, longer window evaluates all of its new points (the reference only looks at its last batch);
  * the same +/- error_buffer padding is used when scoring candidate thresholds and when flagging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

Sequence = tuple[int, int]


@dataclass(frozen=True)
class ThresholdConfig:
    batch_size: int = 70           # values evaluated per step (70 one-minute values)
    window_size: int = 30          # trailing batches per window, so h = 2100
    smoothing_perc: float = 0.05   # EWMA span as a fraction of h
    error_buffer: int = 100        # points padded on each side of a flagged error
    p: float = 0.13                # minimum percent decrease for pruning
    l_s: int = 250                 # model input length (start-of-series handling + index offset)
    z_min: float = 2.5
    z_max: float = 12.0            # exclusive; reference code grid is np.arange(2.5, 12, 0.5)
    z_step: float = 0.5
    max_seqs: int = 5              # guardrail: candidate thresholds may not yield more sequences
    max_anom_frac: float = 0.5     # guardrail: nor flag half the window
    min_error_scale: float = 0.05  # errors must be non-trivial vs. the values (reference "scale check")

    @property
    def h(self) -> int:
        return self.batch_size * self.window_size

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Detection:
    sequences: list[Sequence]      # inclusive ranges in test-array indices (error index + offset)
    scores: list[float]            # telemanom-style severity per sequence
    smoothed: np.ndarray           # smoothed errors, error-index space


# --------------------------------------------------------------------------- small helpers


def runs(mask: np.ndarray) -> list[Sequence]:
    """Inclusive (start, end) ranges of consecutive True values."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Pad every True by `radius` points on each side (clipped to the array)."""
    idx = np.flatnonzero(mask)
    n = len(mask)
    if len(idx) == 0 or radius <= 0:
        return mask.astype(bool).copy()
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, np.maximum(idx - radius, 0), 1)
    np.add.at(diff, np.minimum(idx + radius, n - 1) + 1, -1)
    return np.cumsum(diff[:-1]) > 0


def smooth_errors(errors: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
    """EWMA-smooth errors; replace the first l_s values (not enough history) with an early average."""
    span = max(int(cfg.h * cfg.smoothing_perc), 1)
    e_s = pd.Series(np.asarray(errors, dtype=np.float64)).ewm(span=span).mean().to_numpy(copy=True)
    if len(e_s) > cfg.l_s:
        e_s[: cfg.l_s] = e_s[: 2 * cfg.l_s].mean()
    return e_s


def z_grid(cfg: ThresholdConfig) -> np.ndarray:
    return np.arange(cfg.z_min, cfg.z_max, cfg.z_step)


def num_to_ignore(n_errors: int, cfg: ThresholdConfig) -> int:
    """Initial errors the first window ignores (needs history); fewer for short channels."""
    if n_errors < 1800:
        return 0
    if n_errors < 2500:
        return cfg.l_s
    return 2 * cfg.l_s


def window_bounds(n: int, cfg: ThresholdConfig) -> list[tuple[int, int, int]]:
    """(start, end, new_from) per window. Window 0 judges all its points; later ones only their new tail."""
    h, b = cfg.h, cfg.batch_size
    if n <= h:
        return [(0, n, 0)]
    n_win = (n - h) // b
    bounds = []
    for i in range(n_win + 1):
        start = i * b
        end = n if i == n_win else h + i * b
        new_from = 0 if i == 0 else h + (i - 1) * b
        bounds.append((start, end, new_from))
    return bounds


# --------------------------------------------------------------------------- core steps


def find_epsilon(e: np.ndarray, mean: float, sd: float, cfg: ThresholdConfig) -> float:
    """Threshold that best separates extreme smoothed errors from the rest (no distributional assumption)."""
    default = mean + cfg.z_max * sd  # reference code's fallback (sd_lim = 12)
    if sd <= 0 or mean <= 0:
        return default
    best_score, epsilon = -np.inf, default
    for z in z_grid(cfg):
        eps = mean + z * sd
        above = e >= eps
        if not above.any():
            continue
        below = e[~above]
        anom = dilate(above, cfg.error_buffer)
        n_seqs = sum(1 for s, t in runs(anom) if s != t)
        n_anom = int(anom.sum())
        reduction = (mean - below.mean()) / mean + (sd - below.std()) / sd
        score = reduction / (n_seqs**2 + n_anom)
        if score >= best_score and n_seqs <= cfg.max_seqs and n_anom < len(e) * cfg.max_anom_frac:
            best_score, epsilon = score, eps
    return float(epsilon)


def prune(e: np.ndarray, anom: np.ndarray, non_anom_max: float, p: float) -> np.ndarray:
    """Drop flagged sequences whose peak is not at least `p` (fractionally) above the next-highest peak.

    Peaks are sorted descending with the largest nominal error appended. Every sequence after the last
    step whose percent decrease reaches p is reclassified as nominal.
    """
    seqs = [(s, t) for s, t in runs(anom) if s != t]
    if not seqs:
        return anom
    peaks = np.array([e[s : t + 1].max() for s, t in seqs])
    order = np.argsort(-peaks, kind="stable")
    ladder = np.append(peaks[order], non_anom_max)
    removed: list[int] = []
    for i in range(len(seqs)):
        top = ladder[i]
        decrease = (top - ladder[i + 1]) / top if top > 0 else 0.0
        removed = removed + [int(order[i])] if decrease < p else []
    keep = np.zeros_like(anom, dtype=bool)
    for j, (s, t) in enumerate(seqs):
        if j not in removed:
            keep[s : t + 1] = True
    return anom & keep


def threshold_window(e_s: np.ndarray, y: np.ndarray, prior: np.ndarray, judge_from: int,
                     cfg: ThresholdConfig) -> tuple[np.ndarray, np.ndarray]:
    """Flag anomalous points in one window. Returns (flag mask, per-point score) over the window.

    `prior` marks points already flagged by earlier windows; `judge_from` is the first index that this
    window is allowed to flag.
    """
    flags = np.zeros(len(e_s), dtype=bool)
    score = np.zeros(len(e_s))
    mean, sd = float(e_s.mean()), float(e_s.std())
    inter_range = float(np.subtract(*np.percentile(y, [95, 5])))
    big_enough = sd > cfg.min_error_scale * float(y.std()) or e_s.max() > cfg.min_error_scale * inter_range
    if not big_enough or not e_s.max() > cfg.min_error_scale:
        return flags, score
    for e in (e_s, 2 * mean - e_s):  # regular errors, then errors mirrored about their mean
        eps = find_epsilon(e, mean, sd, cfg)
        anom = dilate((e >= eps) & (e > cfg.min_error_scale * inter_range), cfg.error_buffer)
        anom[:judge_from] = False
        if not anom.any():
            continue
        nominal = ~(anom | prior)
        non_anom_max = float(e[nominal].max()) if nominal.any() else 0.0
        anom = prune(e, anom, non_anom_max, cfg.p)
        flags |= anom
        score = np.where(anom, np.maximum(score, np.abs(e - eps) / (mean + sd)), score)
    return flags, score


def detect(errors: np.ndarray, y_true: np.ndarray, cfg: ThresholdConfig, offset: int | None = None) -> Detection:
    """Run batched dynamic thresholding over a channel's absolute prediction errors.

    `errors[i]` and `y_true[i]` refer to test index i + offset (offset defaults to cfg.l_s).
    """
    errors = np.abs(np.asarray(errors, dtype=np.float64))
    y_true = np.asarray(y_true, dtype=np.float64)
    if errors.shape != y_true.shape:
        raise ValueError(f"errors {errors.shape} vs y_true {y_true.shape}")
    offset = cfg.l_s if offset is None else offset
    n = len(errors)
    e_s = smooth_errors(errors, cfg)
    flagged = np.zeros(n, dtype=bool)
    score_at = np.zeros(n)
    for k, (start, end, new_from) in enumerate(window_bounds(n, cfg)):
        judge_from = num_to_ignore(n, cfg) if k == 0 else new_from - start
        flags, score = threshold_window(e_s[start:end], y_true[start:end], flagged[start:end], judge_from, cfg)
        flagged[start:end] |= flags
        score_at[start:end] = np.maximum(score_at[start:end], score)
    seqs = [(s, t) for s, t in runs(flagged) if s != t]
    return Detection(
        sequences=[(s + offset, t + offset) for s, t in seqs],
        scores=[float(score_at[s : t + 1].max()) for s, t in seqs],
        smoothed=e_s,
    )

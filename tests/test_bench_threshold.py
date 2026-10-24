"""Dynamic thresholding (Hundman et al. 2018) on toy error series."""

import numpy as np

from vbw.bench.threshold import ThresholdConfig, detect, dilate, prune, runs, window_bounds


def _toy(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    errors = 0.02 + 0.01 * rng.random(n)
    y = np.sin(np.linspace(0, 40 * np.pi, n))
    return errors, y


def test_finds_obvious_spike():
    cfg = ThresholdConfig()
    errors, y = _toy()
    errors[2000:2020] = 1.0
    det = detect(errors, y, cfg)
    assert len(det.sequences) == 1
    start, end = det.sequences[0]
    assert start <= 2000 + cfg.l_s and end >= 2019 + cfg.l_s  # reported in test-array indices
    assert det.scores[0] > 0


def test_quiet_series_has_no_detections():
    errors, y = _toy()
    assert detect(errors, y, ThresholdConfig()).sequences == []


def test_finds_spike_in_short_channel():
    cfg = ThresholdConfig()
    errors, y = _toy(n=1200)
    errors[700:710] = 0.8
    det = detect(errors, y, cfg, offset=0)
    assert any(s <= 700 <= e for s, e in det.sequences)


def test_prune_keeps_only_peaks_with_clear_drop():
    e = np.zeros(100)
    anom = np.zeros(100, dtype=bool)
    e[10:13], anom[10:13] = 10.0, True
    e[50:53], anom[50:53] = 2.0, True
    kept = prune(e, anom, non_anom_max=1.9, p=0.13)  # 2.0 -> 1.9 is a 5% drop: reclassified nominal
    assert runs(kept) == [(10, 12)]


def test_prune_keeps_lower_peak_when_later_drop_is_large():
    e = np.zeros(100)
    anom = np.zeros(100, dtype=bool)
    for (s, v) in [(10, 10.0), (40, 5.0), (70, 4.9)]:
        e[s : s + 3], anom[s : s + 3] = v, True
    kept = prune(e, anom, non_anom_max=1.0, p=0.13)  # 4.9 -> 1.0 clears p, so everything above stays
    assert len(runs(kept)) == 3


def test_helpers():
    m = np.zeros(10, dtype=bool)
    m[[2, 3, 7]] = True
    assert runs(m) == [(2, 3), (7, 7)]
    assert runs(dilate(m, 1)) == [(1, 4), (6, 8)]
    assert window_bounds(100, ThresholdConfig()) == [(0, 100, 0)]
    b = window_bounds(2100 + 70 * 2 + 5, ThresholdConfig())
    assert b[0] == (0, 2100, 0) and b[1] == (70, 2170, 2100) and b[-1] == (140, 2245, 2170)

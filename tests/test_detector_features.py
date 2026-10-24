"""Phase balancing of L3 training windows: flight phase alone must not predict the label."""

from __future__ import annotations

import numpy as np

from vbw.detector.features import ForgedSet, flight_phase, phase_balance


def _windows(n_flights: int = 2, per_flight: int = 100, onset: float = 0.6) -> ForgedSet:
    """Like the real training set: an attack file (forged after `onset` of each flight) plus clean twins of every window."""
    t_end = np.tile(np.linspace(1.0, 100.0, per_flight), n_flights)
    scenario = np.repeat([f"f9:{i}" for i in range(n_flights)], per_flight)
    t = np.stack([t_end - 1.0, t_end], axis=1)
    n = len(t_end)

    def part(y, attack):
        return ForgedSet(np.zeros((n, 2, 8), np.float32), t, y.astype(np.int8), np.array([attack] * n), scenario,
                         "falcon9", 2)

    return ForgedSet.concat([part(t_end > onset * 100.0, "drift"), part(np.zeros(n), "clean")])


def test_flight_phase_is_relative_to_each_flight():
    fs = _windows()
    assert np.isclose(flight_phase(fs).max(), 1.0) and flight_phase(fs).min() > 0.0


def test_phase_balance_equalizes_labels_per_bin_and_keeps_early_clean_windows():
    fs = _windows(onset=0.6)
    out = phase_balance(fs, np.random.default_rng(0), bins=10, min_clean=5)
    which = np.minimum((flight_phase(out) * 10).astype(int), 9)
    for b in range(6, 10):                      # late bins: as many clean as forged windows
        assert (out.y[which == b] == 1).sum() == (out.y[which == b] == 0).sum()
    for b in range(0, 5):                       # early bins have no forged windows but keep some clean ones
        assert (out.y[which == b] == 1).sum() == 0 and (out.y[which == b] == 0).sum() == 5

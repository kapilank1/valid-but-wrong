"""Event-level and point-adjusted scoring."""

import numpy as np
import pytest

from vbw.bench.metrics import match_events, point_adjust, point_adjusted_counts, precision_recall_f


def test_event_matching():
    predicted = [(0, 5), (10, 20), (22, 25), (100, 110)]
    labeled = [(3, 4), (15, 30), (50, 60)]
    counts, hit, false_alarm = match_events(predicted, labeled)
    assert hit == [True, True, False]
    assert false_alarm == [False, False, False, True]
    assert (counts.tp, counts.fp, counts.fn) == (2, 1, 1)  # two predictions in one label = one TP


def test_one_prediction_spanning_two_labels_hits_both():
    counts, _, _ = match_events([(0, 100)], [(10, 20), (50, 60)])
    assert (counts.tp, counts.fp, counts.fn) == (2, 0, 0)


def test_no_predictions():
    counts, hit, _ = match_events([], [(1, 2)])
    assert (counts.tp, counts.fp, counts.fn) == (0, 0, 1) and hit == [False]


def test_inclusive_boundaries():
    assert match_events([(5, 9)], [(9, 12)])[0].tp == 1
    assert match_events([(5, 8)], [(9, 12)])[0].tp == 0


def test_precision_recall_f():
    s = precision_recall_f(tp=8, fp=2, fn=8)
    assert s["precision"] == pytest.approx(0.8)
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert s["f0_5"] == pytest.approx(1.25 * 0.8 * 0.5 / (0.25 * 0.8 + 0.5))
    assert precision_recall_f(0, 0, 0)["f1"] == 0.0


def test_point_adjust():
    label = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    pred = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert point_adjust(pred, label).astype(int).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    c = point_adjusted_counts(pred, label)
    assert (c.tp, c.fp, c.fn) == (3, 1, 2)

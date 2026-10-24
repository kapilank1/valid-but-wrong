"""Label parsing, loaders and the small predictor."""

import numpy as np
import pytest

from vbw.bench.data import DATA_DIR, load_labels, parse_classes, parse_sequences, sequences_to_mask, unique_channels
from vbw.bench.model import ModelConfig, fit_and_predict, n_windows, window_view


def test_parse_sequences():
    assert parse_sequences("[[2149, 2349], [4536, 4844]]") == ((2149, 2349), (4536, 4844))
    assert parse_sequences("[]") == ()
    for bad in ("[[5, 1]]", "[[1, 2, 3]]", "__import__('os')", "[[1.5, 2]]"):
        with pytest.raises(ValueError):
            parse_sequences(bad)


def test_parse_classes():
    assert parse_classes("[contextual, point]") == ("contextual", "point")
    assert parse_classes("[point]") == ("point",)


def test_load_labels_from_csv(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text('chan_id,spacecraft,anomaly_sequences,class,num_values\n'
                   'P-1,SMAP,"[[1, 2], [5, 9]]","[point, contextual]",20\n'
                   'P-1,SMAP,"[[3, 4]]",[point],20\n')
    rows = load_labels(csv)
    assert [r.row for r in rows] == [0, 1]
    assert rows[0].sequences == ((1, 2), (5, 9)) and rows[0].classes == ("point", "contextual")
    assert unique_channels(rows) == ["P-1"]


@pytest.mark.skipif(not (DATA_DIR / "labeled_anomalies.csv").exists(), reason="SMAP/MSL not downloaded")
def test_real_label_counts_match_paper():
    rows = load_labels()
    for sc, n_rows, n_anoms in (("SMAP", 55, 69), ("MSL", 27, 36)):
        sub = [r for r in rows if r.spacecraft == sc]
        assert len(sub) == n_rows and sum(len(r.sequences) for r in sub) == n_anoms
    assert all(len(r.sequences) == len(r.classes) for r in rows)


def test_sequences_to_mask():
    assert sequences_to_mask([(1, 2), (4, 10)], 6).astype(int).tolist() == [0, 1, 1, 0, 1, 1]


def test_windows_and_tiny_model():
    cfg = ModelConfig(l_s=12, n_predictions=3, hidden=8, layers=1, epochs=2, batch_size=16)
    rng = np.random.default_rng(0)
    series = np.column_stack([np.sin(np.arange(200) / 5), rng.integers(0, 2, 200)]).astype(np.float32)
    view = window_view(series, cfg)
    assert view.shape == (n_windows(200, cfg), 15, 2)
    np.testing.assert_array_equal(view[7, :, 0], series[7:22, 0])
    pred = fit_and_predict("toy", series[:120], series[120:], cfg, "cpu")
    assert pred.y_hat.shape == pred.y_true.shape == (80 - 15,)
    np.testing.assert_allclose(pred.y_true, series[120 + 12 : 200 - 3, 0], rtol=1e-6)

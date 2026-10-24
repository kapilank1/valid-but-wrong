import numpy as np
import pytest
import torch

from tests.detector_synth import falcon9_track
from vbw.detector.data import Run
from vbw.detector.l2 import L2Model, training_pairs
from vbw.detector.nets import GRUPredictor, export_onnx
from vbw.detector.train import FitConfig, ewma_series, fit_l2_model, fit_norm
from vbw.schema import FALCON9


@pytest.fixture(scope="module")
def tiny_l2(tmp_path_factory):
    runs = [Run("synth", *falcon9_track(noise=1.0, seed=s)[:2], None) for s in range(2)]
    for r in runs:
        r.truth = r.X
    norm = fit_norm(FALCON9, runs)
    model = fit_l2_model(norm, runs[:1], runs[1:], FitConfig(epochs=1, hidden=8, batch=128, log=False))
    path = tmp_path_factory.mktemp("models") / "l2_falcon9.onnx"
    example = training_pairs(runs[0].X[:40], norm)[0][:3]
    export_onnx(model, example, path, "delta")
    return model, L2Model(path, norm), runs


def test_onnx_matches_torch(tiny_l2):
    model, l2, runs = tiny_l2
    inputs = training_pairs(runs[1].X, l2.norm)[0][:64]
    with torch.no_grad():
        expected = model(torch.from_numpy(inputs)).numpy()
    np.testing.assert_allclose(l2.predict(inputs), expected, atol=1e-4)


def test_streaming_z_matches_batched_series(tiny_l2):
    _, l2, runs = tiny_l2
    X = runs[1].X[:120]
    series = l2.zscore_series(X)
    assert not series[: l2.needs - 1].any()
    for k in (l2.needs, 80, 119):
        np.testing.assert_allclose(l2.zscore_last(X[: k + 1]), series[k], atol=1e-5)
    np.testing.assert_allclose(l2.zscore_series(X, last=10)[-10:], series[-10:], atol=1e-6)


def test_step_injection_gives_large_z(tiny_l2):
    _, l2, runs = tiny_l2
    X = runs[1].X[:100].copy()
    clean = np.abs(l2.zscore_last(X)).max()
    X[-1, FALCON9.index("altitude")] += 50 * l2.norm.dstd[FALCON9.index("altitude")]
    assert np.abs(l2.zscore_last(X)).max() > clean + 10


def test_ewma_series_matches_streaming_update():
    z = np.random.default_rng(0).normal(size=(50, 3))
    state, out = np.zeros(3), []
    for row in z:
        state = 0.9 * state + 0.1 * row
        out.append(state)
    np.testing.assert_allclose(ewma_series(z, 0.1), np.array(out))


def test_gru_predictor_shapes():
    net = GRUPredictor(16, 8, hidden=4)
    assert net(torch.zeros(5, 20, 16)).shape == (5, 8)

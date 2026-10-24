import json

import numpy as np
import pytest
import torch

from tests.detector_synth import falcon9_track
from vbw.detector.alerting import AlertConfig
from vbw.detector.base import BaseDetector
from vbw.detector.data import Run
from vbw.detector.features import L3_CONTEXT, feature_names, l3_features, l3_window, load_forged_npz, save_forged_npz
from vbw.detector.l2 import L2Model, model_paths, training_pairs
from vbw.detector.l3 import HardenedDetector, L3Model
from vbw.detector.nets import WindowClassifier, export_onnx
from vbw.detector.physics import FALCON9_RULES, L1Config
from vbw.detector.registry import list_detectors, make_detector
from vbw.detector.spacecraft import SpacecraftDetector
from vbw.detector.train import FitConfig, fit_l2_model, fit_norm
from vbw.schema import FALCON9, QUAD, SPACECRAFT, Frame, L0Result

L1_DEFAULTS = L1Config("falcon9", {r.name: r.default_threshold for r in FALCON9_RULES}, {})
ALERT = AlertConfig(tau=0.5, k=3, n=5, clear_frames=20, warmup=10)


def frames(t, X, schema=FALCON9):
    return [Frame("test", schema.schema_id, i, float(tt), dict(zip(schema.channels, map(float, x))))
            for i, (tt, x) in enumerate(zip(t, X))]


def run(det, fs):
    det.reset()
    return [det.update(f, L0Result()) for f in fs]


def l1_only():
    return BaseDetector(FALCON9, l1_config=L1_DEFAULTS, l2=None, alert=ALERT)


def test_registry_lists_and_builds_every_detector():
    assert [d["name"] for d in list_detectors()] == ["base", "hardened"]
    assert make_detector("base", FALCON9).name == "base"
    assert make_detector("hardened", FALCON9).name == "hardened"
    assert make_detector("base", QUAD).name == "base"
    assert isinstance(make_detector("hardened", SPACECRAFT), SpacecraftDetector)
    with pytest.raises(KeyError):
        make_detector("nope", FALCON9)


def test_clean_track_never_alerts():
    t, X = falcon9_track(noise=1.0)
    verdicts = run(l1_only(), frames(t, X))
    assert not any(v.alert for v in verdicts)
    assert max(v.score for v in verdicts) < 0.5


def test_warmup_frames_never_alert_even_on_garbage():
    t, X = falcon9_track(noise=1.0)
    X[:, FALCON9.index("accel")] = 500.0          # violates the envelope from frame 0
    verdicts = run(l1_only(), frames(t, X))
    assert not any(v.alert for v in verdicts[:ALERT.warmup])
    assert verdicts[ALERT.warmup + ALERT.k].alert
    assert "beyond what Falcon 9 can do" in verdicts[ALERT.warmup + ALERT.k].explanation


def test_bias_alerts_with_explanation_and_clears_after_attack():
    t, X = falcon9_track(noise=1.0)
    bad = X.copy()
    bad[200:260, FALCON9.index("vx")] += 25.0
    verdicts = run(l1_only(), frames(t, bad))
    assert not any(v.alert for v in verdicts[:200])
    first = next(i for i, v in enumerate(verdicts) if v.alert)
    assert first - 200 < 20
    assert "horizontal velocity" in verdicts[first].explanation
    assert not verdicts[-1].alert                 # cleared once the integration window is clean again


def test_corrupted_frames_are_skipped_not_scored():
    t, X = falcon9_track(noise=1.0)
    det = l1_only()
    fs = frames(t, X)
    for f in fs[:150]:
        det.update(f, L0Result())
    bad = fs[150].copy()
    bad.channels["vz"] = 1e9
    v = det.update(bad, L0Result(crc_ok=False, reasons=["CRC mismatch"]))
    assert not v.alert and "CRC mismatch" in v.explanation
    assert not det.update(fs[151], L0Result()).l1_violations


def test_spacecraft_detector_flags_a_step():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.01, 600)
    values[400:] += 0.5
    det = SpacecraftDetector()
    out = [det.update(Frame("sc", SPACECRAFT.schema_id, i, i * 1.0, {"value": float(v)}), L0Result())
           for i, v in enumerate(values)]
    assert not any(v.alert for v in out[:400])
    assert any(v.alert for v in out[400:410])


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    runs = [Run("synth", *falcon9_track(noise=1.0, seed=s), None) for s in range(2)]
    norm = fit_norm(FALCON9, runs)
    model = fit_l2_model(norm, runs[:1], runs[1:], FitConfig(epochs=1, hidden=8, batch=128, log=False))
    onnx_path, norm_path = model_paths(FALCON9, d)
    export_onnx(model, training_pairs(runs[0].X[:40], norm)[0][:2], onnx_path, "delta")
    norm.save(norm_path)
    torch.manual_seed(0)
    clf = WindowClassifier(len(feature_names(FALCON9)), hidden=4)
    export_onnx(clf, np.zeros((2, L3_CONTEXT, len(feature_names(FALCON9))), np.float32), d / "l3_falcon9.onnx", "logit")
    (d / "l3_falcon9_meta.json").write_text(json.dumps({"logit_threshold": 0.0, "context": L3_CONTEXT}))
    return d


def test_hardened_streaming_features_match_offline_features(tiny_models):
    t, X = falcon9_track(noise=1.0)
    l3 = L3Model.load(FALCON9, tiny_models)
    det = HardenedDetector(FALCON9, tiny_models, l3=l3, l1_config=L1_DEFAULTS, alert=ALERT)
    verdicts = run(det, frames(t, X))
    assert verdicts[-1].l3_prob is not None and 0.0 <= verdicts[-1].l3_prob <= 1.0
    W = l3_window(FALCON9)
    offline = l3_features(X[None, -W:], t[None, -W:], FALCON9, L1_DEFAULTS, L2Model.load(FALCON9, tiny_models))[0]
    np.testing.assert_allclose(np.stack(det._features), offline, atol=1e-3)


def test_hardened_without_l3_equals_base(tiny_models):
    t, X = falcon9_track(noise=1.0)
    X[300:, FALCON9.index("q")] *= 1.3
    base = BaseDetector(FALCON9, tiny_models, l1_config=L1_DEFAULTS, alert=ALERT)
    hard = HardenedDetector(FALCON9, tiny_models, l3=None, l1_config=L1_DEFAULTS, alert=ALERT)
    fs = frames(t, X)
    for a, b in zip(run(base, fs), run(hard, fs)):
        assert (a.score, a.alert, a.explanation) == (b.score, b.alert, b.explanation)
        assert b.l3_prob is None and b.detector == "hardened"


def test_forged_npz_roundtrip(tmp_path):
    W, C = l3_window(FALCON9), len(FALCON9.channels)
    X = np.zeros((3, W, C), np.float32)
    path = tmp_path / "forged_falcon9_test.npz"
    save_forged_npz(path, X, np.zeros((3, W)), [0, 1, 1], ["none", "bias", "bias"], ["falcon9:a"] * 3, "falcon9", W)
    fs = load_forged_npz(path)
    assert fs.X.shape == (3, W, C) and fs.y.tolist() == [0, 1, 1] and fs.attack[1] == "bias"
    with pytest.raises(ValueError):
        save_forged_npz(path, X[:, :10], np.zeros((3, 10)), [0, 1, 1], ["a"] * 3, ["s"] * 3, "falcon9", W)

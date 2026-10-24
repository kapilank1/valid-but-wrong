import numpy as np
import pytest

from tests.detector_synth import falcon9_track, quad_track
from vbw.detector.physics import (
    FALCON9_RULES, QUAD_RULES, FrameBuffer, L1Checker, L1Config, evaluate_rule, integrated_mismatch,
)
from vbw.schema import FALCON9, QUAD

F9_DEFAULTS = L1Config("falcon9", {r.name: r.default_threshold for r in FALCON9_RULES}, {})
QUAD_DEFAULTS = L1Config("quad", {r.name: r.default_threshold for r in QUAD_RULES}, {})
F9_RULE = {r.name: r for r in FALCON9_RULES}
Q_RULE = {r.name: r for r in QUAD_RULES}


def corridor_params(t, X):
    alt, spd = X[:, FALCON9.index("altitude")], X[:, FALCON9.index("speed")]
    return {"corridor": {"t": t.tolist(), "altitude_lo": (alt * 0.9).tolist(), "altitude_hi": (alt * 1.1).tolist(),
                         "speed_lo": (spd * 0.9).tolist(), "speed_hi": (spd * 1.1).tolist(),
                         "altitude_floor": 5000.0, "speed_floor": 200.0, "t_all_end": float(t[-1])}}


def fires(rule, X, t, config):
    return evaluate_rule(rule, X, t, config.thresholds[rule.name], config.params)


def test_falcon9_rules_silent_on_consistent_track():
    t, X = falcon9_track(T=800, noise=1.0)   # long enough for every rule, including the 60 s one
    config = L1Config("falcon9", F9_DEFAULTS.thresholds, corridor_params(t, X))
    for rule in FALCON9_RULES:
        series = np.abs(rule.residual(X, t, config.params))
        assert np.nanmax(series) < config.thresholds[rule.name], rule.name


def _edit(X, channel, fn, start=250):
    X = X.copy()
    i = FALCON9.index(channel)
    X[start:, i] = fn(X[start:, i])
    return X


@pytest.mark.parametrize("rule_name, channel, fn", [
    ("vz_vs_altitude_rate", "vz", lambda v: v + 300.0),
    ("vx_vs_downrange_rate", "vx", lambda v: v + 20.0),
    ("accel_vs_speed_rate", "accel", lambda v: v + 20.0),
    ("speed_vs_components", "speed", lambda v: v + 100.0),
    ("angle_vs_velocity", "angle", lambda v: v + 20.0),
    ("q_vs_density", "q", lambda v: v * 1.5),
    ("accel_envelope", "accel", lambda v: v + 80.0),
    ("jerk_envelope", "accel", lambda v: v + 40.0),
    ("corridor", "altitude", lambda v: v * 1.5),
    ("frozen_channel", "downrange", lambda v: np.full_like(v, v[0])),
])
def test_falcon9_rule_fires_on_crafted_violation(rule_name, channel, fn):
    t, X = falcon9_track(noise=1.0)
    config = L1Config("falcon9", F9_DEFAULTS.thresholds, corridor_params(t, X))
    bad = _edit(X, channel, fn)
    # Evaluate where each rule's window has seen enough of the violation (the edit starts at frame 250).
    end = {"jerk_envelope": 258, "frozen_channel": 285, "accel_vs_speed_rate": 340}.get(rule_name, 320)
    violation = fires(F9_RULE[rule_name], bad[:end], t[:end], config)
    assert violation is not None, rule_name
    assert violation.rule == rule_name and violation.message and abs(violation.value) > violation.threshold


def test_60s_altitude_rule_catches_a_slow_walk_off_the_10s_rule_misses():
    t, X = falcon9_track(T=1200, noise=1.0)
    config = L1Config("falcon9", F9_DEFAULTS.thresholds, corridor_params(t, X))
    bad = _edit(X, "altitude", lambda a: a + 30.0 * (np.arange(len(a)) * 0.1), start=300)  # 30 m/s fake climb
    assert fires(F9_RULE["vz_vs_altitude_rate"], bad, t, config) is None          # limit 90 m/s over 10 s
    v = fires(F9_RULE["vz_vs_altitude_rate_60s"], bad, t, config)                  # limit 25 m/s over 60 s
    assert v is not None and v.message.startswith("Over the last minute, altitude has drifted")


def test_violation_message_is_plain_language():
    t, X = falcon9_track()
    bad = _edit(X, "vz", lambda v: v + 300.0)
    v = fires(F9_RULE["vz_vs_altitude_rate"], bad[:300], t[:300], F9_DEFAULTS)
    assert v.message.startswith("Reported vertical velocity disagrees with altitude rate by")
    assert "limit 90 m/s" in v.message


def test_integrated_mismatch_measures_rate_error():
    t = np.arange(200) * 0.1
    level = 5.0 * t                       # level grows at 5 /s
    rate = np.full_like(t, 3.0)           # but the rate channel says 3 /s
    r = integrated_mismatch(level, rate, t, window=50, smooth=5)
    assert np.isnan(r[:50]).all()
    assert r[-1] == pytest.approx(2.0)


def test_rules_vectorize_over_batches():
    t, X = falcon9_track(noise=1.0)
    batch_X, batch_t = np.stack([X[:200], X[100:300]]), np.stack([t[:200], t[100:300]])
    for rule in FALCON9_RULES:
        single = rule.residual(X[100:300], t[100:300], {})
        batched = rule.residual(batch_X, batch_t, {})
        np.testing.assert_allclose(np.nan_to_num(batched[1]), np.nan_to_num(single), rtol=1e-9, atol=1e-6)


def test_quad_rules_silent_on_consistent_track():
    t, X = quad_track()
    for rule in QUAD_RULES:
        series = np.abs(rule.residual(X, t, {}))
        margin = 1.0 if rule.name == "accel_envelope" else 0.2   # the envelope sits at ~1 g in hover
        assert np.nanmax(series) < QUAD_DEFAULTS.thresholds[rule.name] * margin, rule.name


@pytest.mark.parametrize("rule_name, channel, delta", [
    ("pos_vs_velocity", "pn", 30.0),
    ("velocity_vs_accel", "vn", 10.0),
    ("attitude_vs_body_rates", "roll", 0.8),
    ("baro_vs_gps_alt", "alt", 20.0),
    ("groundspeed_vs_velocity", "groundspeed", 5.0),
    ("accel_envelope", "az", -60.0),
])
def test_quad_rule_fires_on_crafted_violation(rule_name, channel, delta):
    t, X = quad_track()
    bad = X.copy()
    bad[200:, QUAD.index(channel)] += delta
    assert fires(Q_RULE[rule_name], bad[:220], t[:220], QUAD_DEFAULTS) is not None


def test_quad_frozen_channel():
    t, X = quad_track()
    bad = X.copy()
    bad[150:, QUAD.index("pe")] = bad[150, QUAD.index("pe")]
    assert fires(Q_RULE["frozen_channel"], bad[:200], t[:200], QUAD_DEFAULTS) is not None


def test_streaming_checker_matches_offline_series():
    t, X = falcon9_track(noise=1.0)
    bad = _edit(X, "vz", lambda v: v + 300.0)
    checker = L1Checker(FALCON9, F9_DEFAULTS)
    results = [checker.update(x, tt) for x, tt in zip(bad, t)]
    series = F9_RULE["vz_vs_altitude_rate"].residual(bad, t, {})
    ratio = series[-1] / F9_DEFAULTS.thresholds["vz_vs_altitude_rate"]
    assert results[-1].ratios["vz_vs_altitude_rate"] == pytest.approx(ratio, rel=1e-6)
    assert any(v.rule == "vz_vs_altitude_rate" for v in results[-1].violations)
    assert not results[50].violations


def test_frame_buffer_keeps_latest_frames_in_order():
    buf = FrameBuffer(2, capacity=5)
    for i in range(23):
        buf.append(np.array([i, -i]), float(i))
    X, t = buf.last(5)
    assert len(buf) == 5 and t.tolist() == [18, 19, 20, 21, 22] and X[-1].tolist() == [22, -22]

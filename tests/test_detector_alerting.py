from vbw.detector.alerting import AlertConfig, alert_series, combine, never_alert_threshold, soft_ratio


def cfg(**kw):
    return AlertConfig(**{"tau": 0.5, "k": 3, "n": 5, "clear_frames": 4, "clear_ratio": 0.8, "warmup": 0, **kw})


def test_soft_ratio_is_half_at_threshold_and_monotonic():
    assert soft_ratio(1.0) == 0.5
    assert soft_ratio(0.0) == 0.0
    assert soft_ratio(0.5) < soft_ratio(1.0) < soft_ratio(2.0) < 1.0


def test_combine_is_noisy_or():
    assert combine(0.0, 0.0) == 0.0
    assert combine(0.5, 0.5) == 0.75
    assert combine(1.0, 0.2) == 1.0


def test_single_spike_never_alerts():
    scores = [0.1] * 10 + [0.99] + [0.1] * 10
    assert not any(alert_series(scores, cfg()))


def test_k_of_n_raises_on_third_suspicious_frame():
    scores = [0.1, 0.9, 0.1, 0.9, 0.9, 0.1]
    assert alert_series(scores, cfg()) == [False, False, False, False, True, True]


def test_hysteresis_holds_through_borderline_and_clears_after_quiet_frames():
    scores = [0.9] * 3 + [0.45] * 6 + [0.1] * 4 + [0.1]
    out = alert_series(scores, cfg())
    assert out[2]                       # raised
    assert all(out[3:9])                # 0.45 is below tau but above clear level 0.4: stays raised
    assert out[9:12] == [True] * 3      # three quiet frames are not enough
    assert out[12] is False             # the fourth clears it


def test_warmup_suppresses_alerts():
    out = alert_series([1.0] * 15, cfg(warmup=10))
    assert not any(out[:12])            # 10 warm-up frames, then k=3 suspicious frames are needed
    assert all(out[12:])


def test_never_alert_threshold_is_the_tightest_tau_with_no_alert():
    scores = [0.1] * 5 + [0.7, 0.2, 0.62, 0.65, 0.1] + [0.1] * 5
    tau = never_alert_threshold(scores, cfg())
    assert tau == 0.62                  # third-largest of the worst 5-frame window
    assert not any(alert_series(scores, cfg(tau=tau)))
    assert any(alert_series(scores, cfg(tau=tau - 0.01)))


def test_never_alert_threshold_ignores_warmup_frames():
    scores = [1.0] * 4 + [0.2] * 10
    assert never_alert_threshold(scores, cfg(warmup=4)) == 0.2

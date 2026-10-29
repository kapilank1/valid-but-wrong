import pytest

from vbw.fleet.metrics import Counter, Gauge, Histogram, render
from vbw.fleet.report import parse, quantile, summarize


def scrape(frames: int, latencies: list[float], vehicles: int = 10, lag: float = 0.0) -> dict:
    h = Histogram("fleet_frame_seconds", "", (0.001, 0.002, 0.005))
    for v in latencies:
        h.observe(v)
    f, vg, lg = Counter("fleet_frames_total", ""), Gauge("fleet_vehicles", ""), Gauge("fleet_lag_seconds", "")
    tf = Gauge("fleet_target_frames_per_second", "")
    f.inc(frames), vg.set(vehicles), lg.set(lag), tf.set(vehicles * 10)
    return parse(render([f, vg, lg, tf, h]))


def test_parse_reads_what_render_writes():
    samples = scrape(42, [0.0015])
    assert samples["fleet_frames_total"] == 42
    assert samples['fleet_frame_seconds_bucket{le="0.002"}'] == 1


def test_quantile_interpolates_within_the_bucket():
    cumulative = [(1.0, 0), (2.0, 100), (float("inf"), 100)]
    assert quantile(0.5, cumulative) == pytest.approx(1.5)
    assert quantile(0.5, [(1.0, 0), (float("inf"), 0)]) is None


def test_summarize_rates_and_fleet_percentiles_from_deltas():
    before = {"worker-0": scrape(1000, [0.0015] * 10), "worker-1": scrape(0, [])}
    after = {"worker-0": scrape(1100, [0.0015] * 10 + [0.0015] * 100),
             "worker-1": scrape(300, [0.004] * 300, lag=3.0)}
    s = summarize(before, after, seconds=10.0)
    assert [p.fps for p in s.pods] == [10.0, 30.0]
    assert s.vehicles == 20 and s.target_fps == 200
    assert not s.keeping_up                              # worker-1 went from 0 to 3 s behind
    assert s.pods[1].lag_trend == pytest.approx(0.3)
    # 100 new observations at 1.5 ms (worker-0) + 300 at 4 ms (worker-1); the earlier 10 are excluded.
    assert 2.0 < s.p50_ms < 5.0 and s.p99_ms <= 5.0


def test_a_pod_draining_its_backlog_counts_as_keeping_up():
    before = {"worker-0": scrape(0, [], lag=40.0)}
    after = {"worker-0": scrape(200, [0.001] * 200, lag=35.0)}
    s = summarize(before, after, seconds=10.0)
    assert s.pods[0].lag_trend == pytest.approx(-0.5)
    assert s.keeping_up

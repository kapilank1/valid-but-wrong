import json
import threading
from datetime import datetime, timezone
import urllib.error
import urllib.request

import pytest

from vbw.fleet import k8s
from vbw.fleet.metrics import Counter, Histogram, render, serve
from vbw.fleet.worker import Worker, WorkerConfig
from vbw.schema import Verdict


class FakeDetector:
    """Never alerts; keeps worker tests fast and independent of the models."""

    name = "fake"

    def reset(self) -> None:
        pass

    def update(self, frame, l0) -> Verdict:
        return Verdict(frame.vehicle_id, frame.seq, frame.t, l0)


def make_worker(ordinal=0, shards=4, fleet=40, members=None):
    config = WorkerConfig(ordinal=ordinal, fleet_size=fleet, attack_fraction=0.2, speed=1000.0)
    return Worker(config, members or (lambda: set(range(shards))), FakeDetector)


def test_workers_together_cover_the_fleet_once():
    workers = [make_worker(ordinal=i) for i in range(4)]
    for w in workers:
        w.maybe_reshard()
    owned = [v for w in workers for v in w.streams]
    assert sorted(owned) == sorted(s.vehicle_id for s in workers[0].specs)


def test_scale_up_drops_vehicles_and_counts_the_move():
    live = {"set": {0}}
    worker = make_worker(members=lambda: live["set"])
    worker.maybe_reshard()
    assert len(worker.streams) == 40 and worker.m_moved.value == 0
    live["set"] = {0, 1, 2, 3}
    worker.maybe_reshard()
    assert 0 < len(worker.streams) < 40
    assert worker.m_moved.value == 40 - len(worker.streams)
    assert worker.m_vehicles.value == len(worker.streams)
    assert worker.m_target_fps.value == len(worker.streams) * 10 * 1000.0


def test_lookup_failure_keeps_current_members():
    calls = {"n": 0}

    def flaky() -> set[int]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("api down")
        return {0, 1}

    worker = make_worker(members=flaky)
    worker.maybe_reshard()
    owned = set(worker.streams)
    worker.maybe_reshard()
    assert set(worker.streams) == owned and worker.shard_members == {0, 1}
    assert worker.m_member_errors.value == 1


def test_a_new_worker_counts_itself_before_the_api_shows_it_ready():
    newcomer = make_worker(ordinal=3, members=lambda: {0, 1, 2})   # not Ready yet, so not listed
    newcomer.maybe_reshard()
    assert newcomer.shard_members == {0, 1, 2, 3}
    assert newcomer.streams                                        # it took its share instead of nothing


def test_survivors_cover_a_missing_workers_vehicles():
    live = {"set": {0, 1, 2, 3}}
    workers = [make_worker(ordinal=i, members=lambda: live["set"]) for i in (0, 1, 3)]
    for w in workers:
        w.maybe_reshard()
    assert sum(len(w.streams) for w in workers) < 40               # worker-2's vehicles belong to worker-2
    live["set"] = {0, 1, 3}                                        # worker-2 is Pending / crashed
    for w in workers:
        w.maybe_reshard()
    assert sum(len(w.streams) for w in workers) == 40              # nobody is left unmonitored


def test_run_scores_one_frame_per_vehicle_per_tick():
    worker = make_worker(shards=1, fleet=10)
    worker.run(threading.Event(), max_ticks=25)
    assert worker.m_frames.value == 250
    assert worker.m_frame_latency.count == 250
    text = worker.metrics_text()
    assert "fleet_frames_total 250" in text and "fleet_vehicles 10" in text


def test_config_from_env_reads_ordinal_from_pod_hostname():
    config = WorkerConfig.from_env({"HOSTNAME": "worker-7", "FLEET_SIZE": "500",
                                    "FLEET_WORKER_SELECTOR": "app=worker"})
    assert (config.ordinal, config.fleet_size, config.worker_selector) == (7, 500, "app=worker")
    assert WorkerConfig.from_env({"HOSTNAME": "x", "FLEET_ORDINAL": "2"}).ordinal == 2


# ---------------------------------------------------------------------- metrics


def test_histogram_buckets_are_cumulative():
    h = Histogram("lat", "latency", (0.001, 0.01))
    for v in (0.0005, 0.005, 0.005, 1.0):
        h.observe(v)
    text = render([h])
    assert 'lat_bucket{le="0.001"} 1' in text
    assert 'lat_bucket{le="0.01"} 3' in text
    assert 'lat_bucket{le="+Inf"} 4' in text
    assert "lat_count 4" in text


def test_metrics_server_reports_readiness():
    ready = {"ok": False}
    counter = Counter("frames_total", "frames")
    server = serve(0, lambda: render([counter]), lambda: ready["ok"])
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/healthz")
        assert err.value.code == 503
        ready["ok"] = True
        assert urllib.request.urlopen(f"{base}/healthz").status == 200
        assert "frames_total 0" in urllib.request.urlopen(f"{base}/metrics").read().decode()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------- kubernetes API


NOW = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)


def pod(name, ready=True, deleting=False, started=None, waiting=False):
    meta = {"name": name, **({"deletionTimestamp": "2026-10-29T11:59:00Z"} if deleting else {})}
    state = {"waiting": {"reason": "CrashLoopBackOff"}} if waiting else (
        {"running": {"startedAt": started}} if started else {})
    return {"metadata": meta, "status": {
        "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        "containerStatuses": [{"ready": ready, "state": state}],
    }}


def test_live_ordinals_rules():
    pods = {"items": [
        pod("worker-0"),                                                   # Ready
        pod("worker-1", ready=False, started="2026-10-29T11:59:50Z"),      # starting: 10 s ago
        pod("worker-2", deleting=True),                                    # terminating
        pod("worker-3", ready=False, started="2026-10-29T11:58:00Z"),      # running 2 min, never Ready: hung
        pod("worker-4", ready=False, waiting=True),                        # crash-looping
        {"metadata": {"name": "worker-5"}, "status": {}},                  # Pending (unschedulable)
    ]}
    assert k8s.live_ordinals(pods, now=NOW) == {0, 1}


def test_live_workers_lists_pods_with_the_service_account(tmp_path):
    (tmp_path / "namespace").write_text("argus\n")
    (tmp_path / "token").write_text("tok123\n")
    seen = {}

    def fetch(request, context):
        seen["url"], seen["auth"] = request.full_url, request.get_header("Authorization")
        return json.loads(json.dumps({"items": [pod("worker-0"), pod("worker-5"), pod("worker-6", ready=False)]}))

    # create_default_context needs a real CA file; any PEM bundle works for building the context.
    import certifi

    (tmp_path / "ca.crt").write_text(open(certifi.where()).read())
    assert k8s.in_cluster(tmp_path)
    assert k8s.live_workers("app.kubernetes.io/name=worker", sa_dir=tmp_path, fetch=fetch) == {0, 5}
    assert seen["url"] == ("https://kubernetes.default.svc/api/v1/namespaces/argus/pods"
                           "?labelSelector=app.kubernetes.io%2Fname%3Dworker")
    assert seen["auth"] == "Bearer tok123"

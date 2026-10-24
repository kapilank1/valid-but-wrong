"""FastAPI endpoints and the /ws feed, against fake registries and the no-op store."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from vbw.pipeline.fakes import FakeWorld
from vbw.pipeline.runner import RunnerConfig
from vbw.pipeline.server import create_app, cors_origins, read_bench
from vbw.store.base import NullStore


@pytest.fixture
def world():
    return FakeWorld(n_frames=400, rate_hz=100.0)


@pytest.fixture
def client(world, tmp_path):
    app = create_app(world.registries(), store=NullStore(), bench_path=tmp_path / "bench_summary.json",
                     runner_config=RunnerConfig(speed_max=1e6))
    with TestClient(app) as c:
        yield c


def wait_until(client, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if predicate(status):
            return status
        time.sleep(0.01)
    raise AssertionError(f"timed out; last status {status}")


def test_health_catalog_status(client):
    assert client.get("/api/health").json()["ok"] is True
    catalog = client.get("/api/catalog").json()
    assert [s["id"] for s in catalog["scenarios"]] == [FakeWorld.SCENARIO]
    assert "bias" in [a["name"] for a in catalog["attacks"]]
    assert {"base", "hardened"} <= {d["name"] for d in catalog["detectors"]}
    status = client.get("/api/status").json()
    expected = {"running", "scenario", "attack", "attack_params", "detector", "speed", "t", "frames", "alerts"}
    assert expected <= set(status)
    assert status["running"] is False and status["scenario"] is None


def test_control_runs_a_scenario_to_the_end(client):
    resp = client.post("/api/control", json={"action": "start", "scenario": FakeWorld.SCENARIO,
                                             "attack": "bias", "speed": 1e6})
    assert resp.status_code == 200 and resp.json()["scenario"] == FakeWorld.SCENARIO
    status = wait_until(client, lambda s: s["finished"])
    assert status["frames"] == 400 and status["alerts"] == 1 and status["attack"] == "bias"


def test_control_rejects_bad_input(client):
    assert client.post("/api/control", json={"action": "fly"}).status_code == 422
    resp = client.post("/api/control", json={"action": "start", "scenario": "nope:x"})
    assert resp.status_code == 400 and "nope:x" in resp.json()["detail"]
    assert client.post("/api/attack", json={"attack": "nope"}).status_code == 400
    assert client.post("/api/detector", json={"detector": "nope"}).status_code == 400


def test_attack_and_detector_hot_swap_endpoints(client):
    client.post("/api/control", json={"action": "reset", "scenario": FakeWorld.SCENARIO})
    resp = client.post("/api/attack", json={"attack": "bias", "attack_params": {"offset": 12.0}})
    assert resp.status_code == 200
    assert resp.json()["attack"] == "bias" and resp.json()["attack_params"]["offset"] == 12.0
    resp = client.post("/api/detector", json={"detector": "hardened"})
    assert resp.status_code == 200 and resp.json()["detector"] == "hardened"


def test_websocket_feed(client):
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["catalog"]["scenarios"][0]["id"] == FakeWorld.SCENARIO
        assert hello["status"]["running"] is False

        client.post("/api/control", json={"action": "start", "scenario": FakeWorld.SCENARIO,
                                          "attack": "bias", "speed": 1e6})
        seen: dict[str, list[dict]] = {}
        for _ in range(2000):
            message = ws.receive_json()
            seen.setdefault(message["type"], []).append(message)
            if message["type"] == "status" and message["status"]["finished"]:
                break
        assert seen["scenario"][0]["scenario"]["id"] == FakeWorld.SCENARIO
        assert len(seen["alert"]) == 1 and seen["alert"][0]["spoken"].startswith("Flight, Guidance.")
        tick = seen["tick"][-1]
        assert tick["verdict"]["l0"]["ok"] is True and tick["attack"] == {"name": "bias", "active": True}

    # A late joiner gets hello followed by the loaded scenario.
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        assert ws.receive_json()["type"] == "scenario"


def test_bench_endpoint(client, tmp_path):
    assert client.get("/api/bench").json() == {"available": False}
    (tmp_path / "bench_summary.json").write_text(json.dumps({"f1": 0.9, "nan_field": None}))
    assert client.get("/api/bench").json() == {"f1": 0.9, "nan_field": None}


def test_read_bench_handles_garbage(tmp_path):
    path = tmp_path / "bench_summary.json"
    path.write_text("{not json")
    assert read_bench(path)["available"] is False


def test_tiger_data_endpoints_without_a_database(client):
    assert client.get("/api/runs").json() == {"available": False}
    assert client.get("/api/runs/0123/timeline").json() == {"available": False}
    assert client.get("/api/analytics").json() == {"available": False}


def test_cors_allows_the_console(client):
    resp = client.options("/api/status", headers={"Origin": "http://localhost:3000",
                                                  "Access-Control-Request-Method": "GET"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert cors_origins({"VBW_CORS_ORIGINS": "https://vbw.example, "})[-1] == "https://vbw.example"

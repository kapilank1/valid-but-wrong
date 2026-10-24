"""Runner tests with fake source/attack/detector (vbw.pipeline.fakes). Each runs in well under a second."""

import asyncio
import time

import pytest

from vbw.pipeline.fakes import FakeWorld
from vbw.pipeline.hub import Hub
from vbw.pipeline.runner import PipelineError, Runner, RunnerConfig, pace_delay

FAST = 1e6  # effectively unpaced


class Recorder:
    def __init__(self):
        self.messages: list[dict] = []

    def __call__(self, message: dict) -> None:
        self.messages.append(message)

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.messages if m["type"] == kind]


def make_runner(world: FakeWorld, publish=None, **config) -> Runner:
    return Runner(world.registries(), publish or Recorder(), config=RunnerConfig(speed_max=FAST, **config))


async def play_to_end(runner: Runner, **kwargs) -> None:
    await runner.control("start", speed=FAST, **kwargs)
    await runner.wait_stopped(timeout=10)


def run(coro):
    return asyncio.run(coro)


def test_pace_delay():
    assert pace_delay(t=11.0, anchor_t=10.0, anchor_wall=100.0, speed=2.0, now=100.0) == pytest.approx(0.5)
    assert pace_delay(t=11.0, anchor_t=10.0, anchor_wall=100.0, speed=2.0, now=101.0) == pytest.approx(-0.5)


def test_plays_every_frame_then_finishes():
    world, rec = FakeWorld(n_frames=300), Recorder()

    async def main():
        runner = make_runner(world, rec)
        await play_to_end(runner, scenario=FakeWorld.SCENARIO)
        return runner

    runner = run(main())
    ticks = rec.of("tick")
    assert len(ticks) == 300
    assert [t["truth"]["seq"] for t in ticks] == list(range(300))
    assert runner.status.frames == 300 and runner.status.finished and not runner.status.running
    assert rec.messages[0]["type"] == "scenario"
    assert rec.messages[0]["scenario"]["id"] == FakeWorld.SCENARIO
    assert rec.of("status")[-1]["status"]["finished"] is True
    assert set(ticks[0]) == {"type", "truth", "reported", "verdict", "attack"}
    assert ticks[0]["verdict"]["l0"]["ok"] is True


def test_attack_passes_l0_while_reported_differs_from_truth():
    world, rec = FakeWorld(n_frames=300), Recorder()
    run(play_to_end(make_runner(world, rec), attack="bias", attack_params={"start_t": 1.0}))
    ticks = rec.of("tick")
    before = [t for t in ticks if t["truth"]["t"] < 1.0]
    after = [t for t in ticks if t["truth"]["t"] >= 1.0]
    assert before and after
    assert all(t["verdict"]["l0"]["ok"] for t in ticks), "a forged frame with a recomputed CRC passes L0"
    assert not any(t["attack"]["active"] for t in before)
    assert all(t["attack"]["active"] and t["attack"]["name"] == "bias" for t in after)
    assert all(t["reported"]["channels"] == t["truth"]["channels"] for t in before)
    assert all(t["reported"]["channels"]["altitude"] == pytest.approx(t["truth"]["channels"]["altitude"] + 50.0)
               for t in after)


def test_alert_rising_edge_fires_exactly_once():
    world, rec = FakeWorld(n_frames=300), Recorder()

    async def main():
        runner = make_runner(world, rec)
        await play_to_end(runner, attack="bias")
        return runner

    runner = run(main())
    alerting = [t for t in rec.of("tick") if t["verdict"]["alert"]]
    alerts = rec.of("alert")
    assert len(alerting) > 50, "the fake detector keeps alerting while the bias persists"
    assert len(alerts) == 1 and runner.status.alerts == 1
    assert alerts[0]["verdict"]["seq"] == alerting[0]["verdict"]["seq"]
    spoken = alerts[0]["spoken"]
    assert spoken.startswith("Flight, Guidance. Altitude disagrees with vertical velocity.")
    assert "Possible spoofed telemetry." in spoken


def test_pacing_follows_speed():
    world = FakeWorld(n_frames=41, rate_hz=100.0)  # 0.4 s of mission time

    async def main(speed):
        runner = make_runner(world)
        start = time.perf_counter()
        await runner.control("start", speed=speed)
        await runner.wait_stopped(timeout=5)
        return time.perf_counter() - start

    assert 0.15 < run(main(2.0)) < 0.6     # 0.4 s mission at 2x -> ~0.2 s
    assert run(main(FAST)) < 0.15


def test_speed_is_clamped_to_contract_range():
    async def main():
        runner = Runner(FakeWorld().registries(), Recorder())
        await runner.control("reset", speed=1000)
        high = runner.status.speed
        await runner.control("reset", speed=0.01)
        return high, runner.status.speed

    assert run(main()) == (20.0, 0.5)


def test_stop_then_start_resumes_without_losing_a_frame():
    world, rec = FakeWorld(n_frames=300, rate_hz=100.0), Recorder()

    async def main():
        runner = make_runner(world, rec)
        await runner.control("start", speed=10.0)   # 3 s of mission at 10x = 0.3 s
        await asyncio.sleep(0.12)
        await runner.control("stop")
        paused_at = runner.status.frames
        await asyncio.sleep(0.05)
        assert runner.status.frames == paused_at and not runner.status.running
        await runner.control("start", speed=FAST)
        await runner.wait_stopped(timeout=5)
        return paused_at, runner

    paused_at, runner = run(main())
    assert 0 < paused_at < 300
    ticks = rec.of("tick")
    assert [t["truth"]["seq"] for t in ticks] == list(range(300))
    assert all(t["verdict"]["l0"]["ok"] for t in ticks), "pausing must not create a sequence gap"
    assert len(rec.of("scenario")) == 1, "resume is not a new run"


def test_reset_starts_a_new_run_from_the_first_frame():
    world, rec = FakeWorld(n_frames=100), Recorder()

    async def main():
        runner = make_runner(world, rec)
        await play_to_end(runner)
        first_run = runner.status.run_id
        await runner.control("reset")
        assert runner.status.frames == 0 and not runner.status.finished and not runner.status.running
        assert runner.status.run_id != first_run
        await play_to_end(runner)
        return runner

    runner = run(main())
    assert runner.status.frames == 100
    assert len(rec.of("scenario")) == 2
    assert len(rec.of("tick")) == 200


def test_hot_swap_detector_is_warmed_and_catches_ongoing_attack():
    world, rec = FakeWorld(n_frames=600, rate_hz=100.0), Recorder()

    async def main():
        runner = make_runner(world, rec, warm_frames=1000)
        await runner.control("start", attack="bias", detector="blind", speed=10.0)
        await asyncio.sleep(0.3)                    # mission t ~ 3 s, the bias began at 1 s
        await runner.control("stop")
        assert not rec.of("alert"), "the blind detector misses the attack"
        before_swap = runner.status.frames
        await runner.set_detector("base")
        await runner.control("start", speed=FAST)
        await runner.wait_stopped(timeout=5)
        return runner, before_swap

    runner, before_swap = run(main())
    assert 100 < before_swap < 600
    fresh = world.detectors[-1]
    assert fresh.name == "base"
    assert fresh.seqs[:before_swap] == list(range(before_swap)), "warmed on every buffered reported frame"
    assert fresh.seqs[before_swap:] == list(range(before_swap, 600))
    alerts = rec.of("alert")
    assert len(alerts) == 1 and alerts[0]["verdict"]["detector"] == "base"
    assert alerts[0]["verdict"]["seq"] == before_swap, "alerts on the first live frame after the swap"
    assert runner.status.detector == "base"


def test_detector_swap_without_warmup_starts_cold():
    world, rec = FakeWorld(n_frames=600, rate_hz=100.0), Recorder()

    async def main():
        runner = make_runner(world, rec, warm_frames=0)
        await runner.control("start", attack="bias", detector="blind", speed=10.0)
        await asyncio.sleep(0.3)
        await runner.control("stop", detector="base")
        await runner.control("start", speed=FAST)
        await runner.wait_stopped(timeout=5)

    run(main())
    # A cold dead-reckoning detector anchors on already-biased altitude and never sees the lie.
    assert not rec.of("alert")


def test_hot_swap_attack_mid_run_starts_now_and_is_warmed():
    world, rec = FakeWorld(n_frames=600, rate_hz=100.0), Recorder()

    async def main():
        runner = make_runner(world, rec, warm_frames=50)
        await runner.control("start", speed=10.0)
        await asyncio.sleep(0.25)
        await runner.control("stop")
        swap_t, swap_frames = runner.status.t, runner.status.frames
        await runner.set_attack("bias", {"offset": 20.0})
        params = dict(runner.status.attack_params)
        await runner.control("start", speed=FAST)
        await runner.wait_stopped(timeout=5)
        return swap_t, swap_frames, params

    swap_t, swap_frames, params = run(main())
    assert params["start_t"] == pytest.approx(swap_t) and params["offset"] == 20.0
    active = [t for t in rec.of("tick") if t["attack"]["active"]]
    assert active[0]["truth"]["seq"] == swap_frames, "the first frame after the swap is forged"
    assert active[0]["attack"]["name"] == "bias"
    attack = world.attacks[-1]
    assert attack.seen == 50 + (600 - swap_frames), "warmed on the last 50 true frames, then live"
    assert len(rec.of("alert")) == 1


def test_loop_survives_attack_and_detector_exceptions():
    world, rec = FakeWorld(n_frames=100), Recorder()

    async def main():
        runner = make_runner(world, rec)
        await play_to_end(runner, attack="boom", detector="flaky")
        return runner

    runner = run(main())
    ticks = rec.of("tick")
    assert runner.status.finished and runner.status.frames == 100
    assert len(ticks) == 90, "the detector raised on every 10th frame; those ticks are skipped"
    assert not any(t["attack"]["active"] for t in ticks), "a broken attack falls back to the clean frame"
    assert all(t["verdict"]["l0"]["ok"] for t in ticks)
    assert runner.status.errors == 100 + 10


def test_bad_requests_raise_pipeline_error():
    async def main():
        runner = make_runner(FakeWorld())
        with pytest.raises(PipelineError):
            await runner.control("start", scenario="nope:nothing")
        with pytest.raises(PipelineError):
            await runner.control("launch")
        await runner.control("reset", scenario=FakeWorld.SCENARIO)
        with pytest.raises(PipelineError):
            await runner.set_attack("does-not-exist")
        with pytest.raises(PipelineError):
            await runner.set_attack("quad_only")          # wrong schema for a falcon9 scenario
        with pytest.raises(PipelineError):
            await runner.set_detector("does-not-exist")
        assert runner.status.attack == "none" and runner.status.detector == "base"

    run(main())


def test_start_without_scenario_uses_catalog_default():
    rec = Recorder()

    async def main():
        runner = make_runner(FakeWorld(n_frames=10), rec)
        await play_to_end(runner)
        return runner

    assert run(main()).status.scenario == FakeWorld.SCENARIO


def test_slow_ws_client_never_stalls_the_loop():
    world = FakeWorld(n_frames=2000, rate_hz=100.0)
    hub = Hub(max_ticks=16)

    async def main():
        stalled = hub.connect()                     # never reads
        runner = Runner(world.registries(), hub.publish, config=RunnerConfig(speed_max=FAST))
        start = time.perf_counter()
        await runner.control("start", attack="bias", speed=FAST)
        await runner.wait_stopped(timeout=10)
        return stalled, runner, time.perf_counter() - start

    stalled, runner, elapsed = run(main())
    assert runner.status.frames == 2000 and elapsed < 5
    texts = [item.text for item in stalled._items]
    kinds = [__import__("json").loads(t)["type"] for t in texts]
    assert kinds.count("tick") == 16 and stalled.dropped == 2000 - 16
    assert kinds.count("alert") == 1 and kinds.count("scenario") == 1 and "status" in kinds

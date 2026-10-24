"""Smoke tests against the real registries. Each one is skipped if a registry it needs doesn't import yet."""

import asyncio
import importlib
import itertools
from dataclasses import dataclass

import pytest

from vbw.pipeline.fakes import DeadReckoningDetector
from vbw.pipeline.registries import Registries
from vbw.pipeline.runner import Runner, RunnerConfig
from vbw.schema import ChannelSchema, Source

N_FRAMES = 300
ATTACK_START_T = 10.0  # 100 clean frames at 10 Hz, then 200 forged ones


def _import(module: str):
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - another builder's module may not exist yet
        pytest.skip(f"{module} not importable yet: {type(exc).__name__}: {exc}")


@dataclass
class Capped:
    """The first `n` frames of a real source."""

    inner: Source
    n: int

    def __post_init__(self):
        self.vehicle_id, self.schema, self.rate_hz = self.inner.vehicle_id, self.inner.schema, self.inner.rate_hz

    def frames(self):
        return itertools.islice(self.inner.frames(), self.n)


class Recorder(list):
    def __call__(self, message):
        self.append(message)

    def of(self, kind):
        return [m for m in self if m["type"] == kind]


def _hero(sources) -> str:
    return next(s["id"] for s in sources.list_scenarios() if s.get("hero"))


def _registries(sources, attacks, make_detector, list_detectors) -> Registries:
    return Registries(
        list_scenarios=sources.list_scenarios,
        make_source=lambda scenario_id, seed=None: Capped(sources.make_source(scenario_id, seed=seed), N_FRAMES),
        list_attacks=attacks.list_attacks,
        make_attack=attacks.make_attack,
        list_detectors=list_detectors,
        make_detector=make_detector,
    )


def _play(registries: Registries, scenario: str, detector: str) -> tuple[Runner, Recorder]:
    rec = Recorder()

    async def main():
        runner = Runner(registries, rec, config=RunnerConfig(speed_max=1e6), seed=0)
        await runner.control("start", scenario=scenario, attack="bias", detector=detector,
                             attack_params={"start_t": ATTACK_START_T}, speed=1e6)
        await runner.wait_stopped(timeout=60)
        return runner

    return asyncio.run(main()), rec


def _assert_valid_but_wrong(runner: Runner, rec: Recorder) -> None:
    ticks = rec.of("tick")
    assert runner.status.finished and runner.status.errors == 0
    assert len(ticks) == N_FRAMES == runner.status.frames
    assert all(t["verdict"]["l0"]["ok"] for t in ticks), "every forged frame passes CRC + sequence checks"
    forged = [t for t in ticks if t["attack"]["active"]]
    assert forged and all(t["truth"]["t"] >= ATTACK_START_T for t in forged)
    assert any(t["reported"]["channels"] != t["truth"]["channels"] for t in forged)
    for message in rec.of("alert"):
        assert message["spoken"].startswith("Flight, Guidance.")


def test_real_hero_launch_and_bias_attack_through_the_link():
    """Real Falcon 9 source and real bias attack; a stand-in detector (the real one may not exist yet)."""
    sources, attacks = _import("vbw.sources.registry"), _import("vbw.attacker.registry")

    def make_detector(name: str, schema: ChannelSchema):
        return DeadReckoningDetector(name=name, threshold=500.0)

    registries = _registries(sources, attacks, make_detector, lambda: [{"name": "base", "label": "", "description": ""}])
    runner, rec = _play(registries, _hero(sources), "base")
    _assert_valid_but_wrong(runner, rec)


def test_full_real_stack_hero_launch_with_bias():
    sources, attacks = _import("vbw.sources.registry"), _import("vbw.attacker.registry")
    detectors = _import("vbw.detector.registry")
    registries = _registries(sources, attacks, detectors.make_detector, detectors.list_detectors)
    runner, rec = _play(registries, _hero(sources), "base")
    _assert_valid_but_wrong(runner, rec)
    assert all(t["verdict"]["detector"] == "base" for t in rec.of("tick"))

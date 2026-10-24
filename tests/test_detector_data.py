"""Clean-run splits: quad runs only fly simulated missions, and val / test never share a noise realization."""

from __future__ import annotations

from types import SimpleNamespace

import vbw.sources.registry as sources_registry
from vbw.detector import data
from vbw.schema import QUAD, Frame


def test_quad_scenarios_exclude_live_sources():
    ids = data.quad_scenarios()
    assert ids
    assert all(s.startswith("quad:") for s in ids)  # sitl:udp:<port> also has the quad schema but blocks on a socket


def test_quad_val_and_test_fly_the_held_mission_on_disjoint_seeds(monkeypatch):
    calls: list[tuple[str, int]] = []

    def fake_make_source(scenario, seed=None):
        calls.append((scenario, seed))
        frame = Frame("q", QUAD.schema_id, 0, 0.0, {c: 0.0 for c in QUAD.channels})
        return SimpleNamespace(frames=lambda: iter([frame]))

    monkeypatch.setattr(sources_registry, "make_source", fake_make_source)
    val = data.quad_runs("val", (0, 1))
    n_val = len(calls)
    test = data.quad_runs("test", (0, 1))
    train = data.quad_runs("train", (0,))

    assert val and {r.scenario for r in val} == {r.scenario for r in test}
    assert not {s for _, s in calls[:n_val]} & {s for _, s in calls[n_val:n_val + len(test)]}
    assert not {r.scenario for r in train} & {r.scenario for r in val}

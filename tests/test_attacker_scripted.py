"""Scripted-attack invariants: seq/t preserved, forged frames pass L0, determinism, start_t gating."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbw.attacker.registry import list_attacks, make_attack
from vbw.link.codec import LinkMonitor, encode
from vbw.schema import FALCON9, Frame

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SCRIPTED_NAMES = ("none", "bias", "drift", "replay", "freeze", "coordinated")


def _frames(n: int = 1200, slug: str = "ses-10") -> list[Frame]:
    df = pd.read_csv(ROOT / "data" / "processed" / "falcon9" / f"{slug}.csv").iloc[:n]
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        channels = {c: float(getattr(r, c)) for c in FALCON9.channels}
        out.append(Frame("f9", FALCON9.schema_id, i & 0xFFFF, float(r.t), channels))
    return out


@pytest.mark.parametrize("name", SCRIPTED_NAMES)
def test_preserves_seq_t_identity(name):
    frames = _frames()
    start_t = 30.0
    atk = make_attack(name, FALCON9, start_t=start_t, intensity=0.7)
    atk.reset(0)
    for f in frames:
        g = atk.apply(f)
        assert (g.seq, g.t, g.vehicle_id, g.schema_id) == (f.seq, f.t, f.vehicle_id, f.schema_id)
        if f.t < start_t:  # identity before the attack begins
            assert g.channels == f.channels


@pytest.mark.parametrize("name", SCRIPTED_NAMES)
def test_forged_frame_passes_l0(name):
    """The core premise: re-encoded forged frames pass every LinkMonitor L0 check."""
    frames = _frames()
    atk = make_attack(name, FALCON9, start_t=20.0, intensity=0.8)
    atk.reset(0)
    mon = LinkMonitor("f9")
    mon.reset()
    for f in frames:
        g = atk.apply(f)
        _, l0 = mon.receive(encode(g))
        assert l0.ok


@pytest.mark.parametrize("name", SCRIPTED_NAMES)
def test_deterministic_with_seed(name):
    f1, f2 = _frames(), _frames()
    a = make_attack(name, FALCON9, start_t=20.0, intensity=0.6)
    b = make_attack(name, FALCON9, start_t=20.0, intensity=0.6)
    a.reset(42)
    b.reset(42)
    for x, y in zip(f1, f2):
        assert a.apply(x).channels == b.apply(y).channels


def test_bias_actually_lies_after_start():
    frames = _frames()
    atk = make_attack("bias", FALCON9, start_t=30.0, intensity=1.0)
    atk.reset(0)
    lied = False
    for f in frames:
        g = atk.apply(f)
        if f.t >= 30.0 and abs(g.channels["altitude"] - f.channels["altitude"]) > 1.0:
            lied = True
    assert lied


def test_coordinated_keeps_speed_hypot_consistent():
    """Coordinated must keep speed == hypot(vx, vz) so naive cross-channel checks pass."""
    frames = _frames()
    atk = make_attack("coordinated", FALCON9, start_t=30.0, intensity=1.0)
    atk.reset(0)
    resid = []
    for f in frames:
        g = atk.apply(f)
        if f.t >= 40.0:
            resid.append(abs(np.hypot(g.channels["vx"], g.channels["vz"]) - g.channels["speed"]))
    assert max(resid) < 1e-3


def test_registry_lists_ppo_and_none():
    names = {a["name"] for a in list_attacks()}
    assert {"none", "ppo"} <= names
    ppo = next(a for a in list_attacks() if a["name"] == "ppo")
    assert "available" in ppo

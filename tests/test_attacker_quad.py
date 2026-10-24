"""Quad-schema attacks on synthetic frames (the real QuadSim source is built by another agent)."""

from __future__ import annotations

import math

import numpy as np

from vbw.attacker.registry import make_attack
from vbw.link.codec import LinkMonitor, encode
from vbw.schema import QUAD, Frame


def _quad_frames(n: int = 300) -> list[Frame]:
    """A gentle climbing hover: position integrates velocity so the truth is self-consistent."""
    out = []
    pn = pe = alt = 0.0
    vn, ve, vu = 1.0, 0.5, 2.0
    dt = 0.05
    for i in range(n):
        pn += vn * dt
        pe += ve * dt
        alt += vu * dt
        ch = {c: 0.0 for c in QUAD.channels}
        ch.update(pn=pn, pe=pe, alt=alt, vn=vn, ve=ve, vu=vu,
                  baro_alt=alt, groundspeed=math.hypot(vn, ve), battery_v=16.0, az=-9.81)
        out.append(Frame("quad-1", QUAD.schema_id, i & 0xFFFF, i * dt, ch))
    return out


def test_quad_attacks_preserve_seq_and_pass_l0():
    for name in ("none", "bias", "drift", "replay", "freeze", "coordinated"):
        atk = make_attack(name, QUAD, start_t=2.0, intensity=0.8)
        atk.reset(0)
        mon = LinkMonitor("quad-1")
        mon.reset()
        for f in _quad_frames():
            g = atk.apply(f)
            assert (g.seq, g.t) == (f.seq, f.t)
            _, l0 = mon.receive(encode(g))
            assert l0.ok


def test_quad_coordinated_keeps_velocity_vs_position_consistent():
    """Coordinated must keep vu == d(alt)/dt so a naive position/velocity check still passes."""
    atk = make_attack("coordinated", QUAD, start_t=2.0, intensity=1.0)
    atk.reset(0)
    prev = None
    resid = []
    for f in _quad_frames():
        g = atk.apply(f)
        if prev is not None and f.t >= 3.0:
            dt = g.t - prev.t
            dalt = (g.channels["alt"] - prev.channels["alt"]) / dt
            resid.append(abs(dalt - g.channels["vu"]))
        prev = g.copy()
    assert max(resid) < 1e-3


def test_quad_coordinated_actually_moves_altitude():
    atk = make_attack("coordinated", QUAD, start_t=2.0, intensity=1.0)
    atk.reset(0)
    moved = False
    for f in _quad_frames():
        g = atk.apply(f)
        if f.t >= 5.0 and abs(g.channels["alt"] - f.channels["alt"]) > 1.0:
            moved = True
    assert moved

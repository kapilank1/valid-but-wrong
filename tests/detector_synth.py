"""Physically consistent synthetic trajectories for detector tests (not real data)."""

from __future__ import annotations

import numpy as np

from vbw.detector.physics import G0, R_EARTH, body_to_ned, cumtrapz, predicted_q
from vbw.schema import FALCON9, QUAD


def falcon9_track(T: int = 400, dt: float = 0.1, noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ascent with a slowly pitching flight-path angle; every F9 rule's relation holds exactly (before noise)."""
    t = np.arange(T) * dt
    gamma = np.radians(80.0 - 0.1 * t)
    a_kin = 15.0 + 0.02 * t
    speed = 50.0 + cumtrapz(a_kin, t)
    vz, vx = speed * np.sin(gamma), speed * np.cos(gamma)
    alt = 1000.0 + cumtrapz(vz, t)
    dr = cumtrapz(vx, t)
    g = G0 * (R_EARTH / (R_EARTH + alt)) ** 2
    X = np.zeros((T, len(FALCON9.channels)))
    for name, col in (("speed", speed), ("altitude", alt), ("vz", vz), ("vx", vx), ("downrange", dr),
                      ("accel", a_kin + g * np.sin(gamma)), ("angle", np.degrees(gamma))):
        X[:, FALCON9.index(name)] = col
    X[:, FALCON9.index("q")] = predicted_q(X, {})
    if noise:
        X += np.random.default_rng(seed).normal(0, noise, X.shape) * np.array([0.1, 0.5, 0.1, 0.1, 1, 0.02, 0.02, 1])
    return t, X


def quad_track(T: int = 300, dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Figure-eight-ish flight with gentle attitude motion; body rates and specific force are derived exactly."""
    t = np.arange(T) * dt
    w = 0.5
    pn, pe, alt = 10 * np.sin(w * t), 5 * np.sin(2 * w * t), 20 + 0.5 * t
    vn, ve, vu = 10 * w * np.cos(w * t), 10 * w * np.cos(2 * w * t), 0.5 * np.ones(T)
    an, ae, au = -10 * w * w * np.sin(w * t), -20 * w * w * np.sin(2 * w * t), np.zeros(T)
    roll, pitch, yaw = 0.1 * np.sin(t), 0.08 * np.cos(0.7 * t), 0.3 * t
    droll, dpitch, dyaw = 0.1 * np.cos(t), -0.056 * np.sin(0.7 * t), 0.3 * np.ones(T)
    p = droll - dyaw * np.sin(pitch)
    q = dpitch * np.cos(roll) + dyaw * np.sin(roll) * np.cos(pitch)
    r = -dpitch * np.sin(roll) + dyaw * np.cos(roll) * np.cos(pitch)
    a_ned = np.stack([an, ae, -au], -1)
    a_ned[:, 2] -= G0                                   # specific force = a - g (NED, z down)
    f_body = np.einsum("tji,tj->ti", body_to_ned(roll, pitch, yaw), a_ned)   # R^T (a - g)
    cols = {"pn": pn, "pe": pe, "alt": alt, "vn": vn, "ve": ve, "vu": vu, "roll": roll, "pitch": pitch, "yaw": yaw,
            "p": p, "q": q, "r": r, "ax": f_body[:, 0], "ay": f_body[:, 1], "az": f_body[:, 2],
            "baro_alt": alt + 0.3, "groundspeed": np.hypot(vn, ve), "battery_v": 16.0 - 0.001 * t}
    X = np.stack([cols[c] for c in QUAD.channels], -1)
    X += np.random.default_rng(1).normal(0, 1e-4, X.shape)  # real sensors jitter; keeps the frozen rule quiet
    return t, X

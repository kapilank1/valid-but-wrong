"""SpoofEnv must satisfy the Gymnasium API (env_checker) on a stub detector, and run episodes."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbw.attacker.env import SpoofEnv, SpoofEnvConfig, StubDetector, make_spoof_env
from vbw.schema import FALCON9, Verdict


def _synthetic_trajectory(T: int = 400):
    t = np.arange(T) * 0.1
    x = np.zeros((T, len(FALCON9.channels)), dtype=float)
    x[:, FALCON9.index("altitude")] = np.linspace(50, 160_000, T)
    x[:, FALCON9.index("vz")] = np.gradient(x[:, FALCON9.index("altitude")], 0.1)
    x[:, FALCON9.index("speed")] = np.linspace(0, 7000, T)
    x[:, FALCON9.index("vx")] = np.linspace(0, 6900, T)
    x[:, FALCON9.index("downrange")] = np.linspace(0, 1.3e6, T)
    return [(t, x)]


def test_env_checker_passes_on_stub():
    env = SpoofEnv(_synthetic_trajectory(), lambda: StubDetector(FALCON9), FALCON9,
                   config=SpoofEnvConfig(episode_len=80))
    check_env(env, skip_render_check=True)


def test_reset_is_deterministic():
    env = SpoofEnv(_synthetic_trajectory(), lambda: StubDetector(FALCON9), FALCON9)
    o1, _ = env.reset(seed=7)
    o2, _ = env.reset(seed=7)
    assert np.allclose(o1, o2)


def test_episode_runs_and_offsets_grow_picture_error():
    env = SpoofEnv(_synthetic_trajectory(), lambda: StubDetector(FALCON9), FALCON9,
                   config=SpoofEnvConfig(episode_len=50, terminate_on_alert=False))
    env.reset(seed=0)
    # push altitude offset up every step; picture error must end positive
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    last_err = 0.0
    for _ in range(50):
        _, _, term, trunc, info = env.step(action)
        last_err = info["picture_error"]
        if term or trunc:
            break
    assert last_err > 0.0


class _FixedScore:
    """Detector stand-in that always returns the same score (and alert)."""

    def __init__(self, score: float, alert: bool = False):
        self.score, self.alert = score, alert

    def reset(self) -> None:
        pass

    def update(self, frame, l0):
        return Verdict(frame.vehicle_id, frame.seq, frame.t, l0, score=self.score, alert=self.alert)


def _first_step_reward(detector, cfg):
    env = SpoofEnv(_synthetic_trajectory(), lambda: detector, FALCON9, config=cfg)
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    return reward, terminated


def test_reward_only_charges_score_above_the_margin_and_penalises_alerts():
    cfg = SpoofEnvConfig(episode_len=10, warmup=0)
    assert _first_step_reward(_FixedScore(cfg.score_margin - 0.1), cfg) == (0.0, False)
    reward, _ = _first_step_reward(_FixedScore(0.5), cfg)
    assert reward == pytest.approx(-cfg.lam * (0.5 - cfg.score_margin))
    reward, terminated = _first_step_reward(_FixedScore(0.9, alert=True), cfg)
    assert terminated
    assert reward == pytest.approx(-cfg.lam * (0.9 - cfg.score_margin) - cfg.alert_penalty)


def test_make_spoof_env_uses_real_launches():
    env = make_spoof_env("falcon9", "base", config=SpoofEnvConfig(episode_len=100))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert len(env.trajectories) > 0


def test_ppo_attack_without_model_is_identity(tmp_path):
    from vbw.attacker.registry import make_attack
    from vbw.schema import Frame

    atk = make_attack("ppo", FALCON9, start_t=0.0, model_path=str(tmp_path / "missing.zip"))
    assert atk.loaded is False
    atk.reset(0)
    t, x = _synthetic_trajectory(20)[0]
    for i in range(20):
        f = Frame("f9", FALCON9.schema_id, i, float(t[i]), {c: float(x[i, j]) for j, c in enumerate(FALCON9.channels)})
        g = atk.apply(f)
        assert g.channels == f.channels and (g.seq, g.t) == (f.seq, f.t)

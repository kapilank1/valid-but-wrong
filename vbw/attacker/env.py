"""Gymnasium environment for the learned attacker, plus the runtime ``PpoAttack`` that wraps a policy.

The attacker's job: steer the *operator's picture* of the vehicle as far from the truth as possible
while keeping the ground detector's score below its alert threshold. At each step the policy nudges a
small set of per-channel offsets; the reported frame is ``truth + offsets``. Reward is the growth in
picture error minus the detector score minus a large penalty (and termination) if an alert fires.

Design notes:
- The env never touches the codec. Forged frames pass L0 by construction, so we build an all-OK
  ``L0Result`` directly and keep ``step()`` cheap enough for PPO's millions of steps.
- :class:`SpoofSpec` holds the observation/offset/normalisation layout and is shared by the env and
  :class:`PpoAttack`, so the policy sees the exact same observation at train and run time.
- :class:`PpoAttack` keeps its own *shadow* detector so, at runtime, it reacts to the same score the
  ground station computes from the forged frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from vbw.schema import ChannelSchema, Frame, L0Result, SCHEMAS_BY_NAME, Verdict
from vbw.attacker.scripted import _Attack

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

Trajectory = tuple[np.ndarray, np.ndarray]  # (t [T], X [T, C]) in schema channel order
DetectorFactory = Callable[[], object]       # () -> object with reset() and update(frame, l0) -> Verdict


# ------------------------------------------------------------------ observation / offset spec


@dataclass(frozen=True)
class SpoofSpec:
    """Shared layout so train-time and run-time observations match exactly."""

    schema_name: str
    target_channels: tuple[str, ...]   # channels the policy can offset
    step_bound: tuple[float, ...]      # max per-step increment per target channel
    picture_channels: tuple[str, ...]  # channels that define "operator picture error"
    norm: dict[str, float]             # per-channel normalisation scale

    @property
    def k(self) -> int:
        return len(self.target_channels)

    def normalize_state(self, schema: ChannelSchema, values: np.ndarray) -> np.ndarray:
        scales = np.array([self.norm.get(c, 1.0) for c in schema.channels], dtype=np.float32)
        return (values.astype(np.float32) / scales).clip(-10.0, 10.0)

    def normalize_offsets(self, offsets: np.ndarray) -> np.ndarray:
        scales = np.array([self.norm.get(c, 1.0) for c in self.target_channels], dtype=np.float32)
        return (offsets.astype(np.float32) / scales).clip(-10.0, 10.0)

    def picture_error(self, offsets: dict[str, float]) -> float:
        return float(sum(abs(offsets.get(c, 0.0)) / self.norm.get(c, 1.0) for c in self.picture_channels))


_F9_NORM = {"altitude": 170000.0, "speed": 7000.0, "vz": 1000.0, "vx": 7000.0,
            "downrange": 1_400_000.0, "accel": 40.0, "angle": 90.0, "q": 30000.0}
_QUAD_NORM = {"pn": 100.0, "pe": 100.0, "alt": 50.0, "vn": 20.0, "ve": 20.0, "vu": 20.0,
              "baro_alt": 50.0, "groundspeed": 20.0}

SPECS: dict[str, SpoofSpec] = {
    "falcon9": SpoofSpec(
        schema_name="falcon9",
        target_channels=("altitude", "vz", "speed", "downrange", "vx"),
        # per-step (0.1 s) increments: a full-scale action injects ~100 m/s of fake altitude/downrange
        # rate and 10 m/s^2 of fake velocity drift; large enough to lie, small enough to be learnable.
        step_bound=(10.0, 1.0, 1.0, 10.0, 1.0),
        picture_channels=("altitude", "downrange"),
        norm=_F9_NORM,
    ),
    "quad": SpoofSpec(
        schema_name="quad",
        target_channels=("alt", "vu", "pn", "vn", "pe"),
        step_bound=(0.5, 0.1, 0.5, 0.1, 0.5),
        picture_channels=("alt", "pn", "pe"),
        norm=_QUAD_NORM,
    ),
}


# ------------------------------------------------------------------ fallback stub detector


@dataclass
class StubDetector:
    """A crude streaming L1 used when the real detector isn't importable yet (training stub + tests).

    It scores a handful of per-channel physics inconsistencies for falcon9 (vz vs d(alt)/dt, speed vs
    hypot, downrange rate vs vx) and a flat zero otherwise. It is good enough to give PPO a gradient and
    to drive the env_checker; it is NOT the real detector and training against it is never saved as a
    production model.
    """

    schema: ChannelSchema
    name: str = "stub"
    alert_threshold: float = 0.5
    persistence: int = 8   # consecutive high-score frames before alerting (kills clean-data transients)

    # Normalisers chosen from the clean train-set residual distributions so clean truth scores < 0.5:
    # the vz/dalt check is noisy around stage events (hence the loose scale), speed/hypot and
    # downrange-rate are tight. Measured p99.9 of clean residuals: vz 356, speed 24, downrange 1.65.
    _NORM = {"vz": 1200.0, "sp": 80.0, "dr": 50.0}

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last: Frame | None = None
        self._run = 0  # consecutive frames at/above threshold

    def update(self, frame: Frame, l0: L0Result) -> Verdict:
        score = 0.0
        hard = 0.0   # drives alerts: only the robust checks, so clean truth never alerts
        ch = frame.channels
        if self.schema.name == "falcon9" and self._last is not None:
            dt = max(frame.t - self._last.t, 1e-3)
            dalt = (ch["altitude"] - self._last.channels["altitude"]) / dt
            ddr = (ch["downrange"] - self._last.channels["downrange"]) / dt
            r_vz = abs(dalt - ch["vz"]) / self._NORM["vz"]          # noisy near stage events; soft only
            r_sp = abs((ch["vx"] ** 2 + ch["vz"] ** 2) ** 0.5 - ch["speed"]) / self._NORM["sp"]
            r_dr = abs(ddr - ch["vx"]) / self._NORM["dr"]
            hard = float(np.clip(max(r_sp, r_dr), 0.0, 1.0))
            score = hard
        self._last = frame.copy()
        self._run = self._run + 1 if hard >= self.alert_threshold else 0
        v = Verdict(vehicle_id=frame.vehicle_id, seq=frame.seq, t=frame.t, l0=l0)
        # Report the alert-driving (robust) score: a naive derivative detector cannot see a constant
        # altitude lie, so the attacker is rewarded for walking altitude off while keeping the
        # cross-channel derivatives consistent. The learned detector is what should catch that drift.
        v.l1_score = score
        v.score = score
        v.alert = self._run >= self.persistence
        v.detector = self.name
        return v


# ------------------------------------------------------------------ environment


@dataclass
class SpoofEnvConfig:
    episode_len: int = 400
    picture_gain: float = 50.0     # scales the (normalized) picture-error reward to compete with penalties
    lam: float = 0.5               # detector-score penalty weight
    alert_penalty: float = 10.0    # one-off penalty when an alert fires
    terminate_on_alert: bool = True


class SpoofEnv(gym.Env):
    """RL env: nudge per-channel offsets to grow the operator-picture lie without tripping the detector."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        trajectories: list[Trajectory],
        detector_factory: DetectorFactory,
        schema: ChannelSchema,
        spec: SpoofSpec | None = None,
        config: SpoofEnvConfig | None = None,
    ) -> None:
        super().__init__()
        if not trajectories:
            raise ValueError("SpoofEnv needs at least one trajectory")
        self.trajectories = trajectories
        self.detector_factory = detector_factory
        self.schema = schema
        self.spoof_spec = spec or SPECS[schema.name]
        self.cfg = config or SpoofEnvConfig()
        self._step_bound = np.array(self.spoof_spec.step_bound, dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.spoof_spec.k,), dtype=np.float32)
        obs_dim = len(schema.channels) + self.spoof_spec.k + 2
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._reset_episode(0, 0)

    def _reset_episode(self, traj_idx: int, start: int) -> None:
        self._t_arr, self._x_arr = self.trajectories[traj_idx]
        self._start = start
        self._i = 0
        self._offsets = np.zeros(self.spoof_spec.k, dtype=np.float32)
        self._last_score = 0.0
        self._last_alert = False
        self._prev_err = 0.0
        self.detector = self.detector_factory()
        self.detector.reset()

    def _obs(self) -> np.ndarray:
        row = self._x_arr[self._start + self._i]
        state = self.spoof_spec.normalize_state(self.schema, row)
        offs = self.spoof_spec.normalize_offsets(self._offsets)
        tail = np.array([self._last_score, 1.0 if self._last_alert else 0.0], dtype=np.float32)
        return np.concatenate([state, offs, tail]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        traj_idx = int(self._rng.integers(len(self.trajectories)))
        t_arr, _ = self.trajectories[traj_idx]
        max_start = max(len(t_arr) - self.cfg.episode_len - 1, 0)
        start = int(self._rng.integers(max_start + 1))
        self._reset_episode(traj_idx, start)
        return self._obs(), {}

    def _reported_frame(self) -> tuple[Frame, dict[str, float]]:
        idx = self._start + self._i
        row = self._x_arr[idx]
        channels = {c: float(row[j]) for j, c in enumerate(self.schema.channels)}
        off_map: dict[str, float] = {}
        for k, ch in enumerate(self.spoof_spec.target_channels):
            channels[ch] += float(self._offsets[k])
            off_map[ch] = float(self._offsets[k])
        frame = Frame("rl", self.schema.schema_id, idx & 0xFFFF, float(self._t_arr[idx]), channels)
        return frame, off_map

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._offsets = self._offsets + action * self._step_bound

        frame, off_map = self._reported_frame()
        verdict = self.detector.update(frame, L0Result())
        self._last_score = float(getattr(verdict, "score", 0.0))
        self._last_alert = bool(getattr(verdict, "alert", False))

        err = self.spoof_spec.picture_error(off_map)
        reward = self.cfg.picture_gain * (err - self._prev_err) - self.cfg.lam * self._last_score
        self._prev_err = err

        terminated = False
        if self._last_alert:
            reward -= self.cfg.alert_penalty
            terminated = self.cfg.terminate_on_alert

        self._i += 1
        truncated = (self._i >= self.cfg.episode_len) or (self._start + self._i >= len(self._t_arr))
        info = {"picture_error": err, "score": self._last_score, "alert": self._last_alert}
        return self._obs(), float(reward), terminated, truncated, info


# ------------------------------------------------------------------ runtime attack


@dataclass
class PpoAttack(_Attack):
    """Runtime attack driven by a trained PPO policy against a shadow copy of the ground detector.

    Falls back to a no-op (identity) if no model is on disk, so the pipeline never crashes when PPO
    hasn't been trained yet; ``loaded`` reports whether a real policy is driving it.
    """

    name = "ppo"
    model_path: str | None = None
    detector_name: str = "base"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.spec = SPECS.get(self.schema.name)
        self._step_bound = np.array(self.spec.step_bound, dtype=np.float32) if self.spec else None
        self._policy = self._load_policy()
        self.loaded = self._policy is not None
        self._shadow = _make_shadow(self.detector_name, self.schema)
        self._reset_state()

    def _load_policy(self):
        path = Path(self.model_path) if self.model_path else MODELS / f"ppo_{self.schema.name}.zip"
        if not path.exists() or self.spec is None:
            return None
        try:
            from stable_baselines3 import PPO

            return PPO.load(str(path), device="cpu")
        except Exception:  # missing SB3 or a bad model shouldn't take the pipeline down
            return None

    def _reset_state(self) -> None:
        self._offsets = np.zeros(self.spec.k, dtype=np.float32) if self.spec else np.zeros(0)
        self._last_score = 0.0
        self._last_alert = False
        if self._shadow is not None:
            self._shadow.reset()

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._reset_state()

    def _obs(self, frame: Frame) -> np.ndarray:
        row = np.array([frame.channels[c] for c in self.schema.channels], dtype=np.float32)
        state = self.spec.normalize_state(self.schema, row)
        offs = self.spec.normalize_offsets(self._offsets)
        tail = np.array([self._last_score, 1.0 if self._last_alert else 0.0], dtype=np.float32)
        return np.concatenate([state, offs, tail]).astype(np.float32)

    def apply(self, frame: Frame) -> Frame:
        return self.__call__(frame)

    def __call__(self, frame: Frame) -> Frame:
        if not self._active(frame) or self._policy is None or self.spec is None:
            self._last = frame.copy()
            return frame.copy()
        action, _ = self._policy.predict(self._obs(frame), deterministic=True)
        self._offsets = self._offsets + np.clip(action, -1.0, 1.0).astype(np.float32) * self._step_bound
        out = frame.copy()
        for k, ch in enumerate(self.spec.target_channels):
            out.channels[ch] = out.channels[ch] + float(self._offsets[k])
        if self._shadow is not None:  # react to the same score the ground sees
            v = self._shadow.update(out.copy(), L0Result())
            self._last_score = float(getattr(v, "score", 0.0))
            self._last_alert = bool(getattr(v, "alert", False))
        self._last = frame.copy()
        return out


def _make_shadow(detector_name: str, schema: ChannelSchema):
    try:
        from vbw.detector.registry import make_detector

        return make_detector(detector_name, schema)
    except Exception:
        return StubDetector(schema)


# ------------------------------------------------------------------ training helpers


def load_training_trajectories(schema_name: str = "falcon9", holdout: bool = False) -> list[Trajectory]:
    """Load clean (t, X) arrays for the train (or holdout) launches in schema channel order."""
    from vbw.detector.splits import holdout_slugs, load_falcon9, train_slugs

    slugs = holdout_slugs() if holdout else train_slugs()
    return [load_falcon9(s) for s in slugs]


def make_detector_factory(detector_name: str, schema: ChannelSchema) -> DetectorFactory:
    """A factory that builds a fresh detector per episode, falling back to the stub if unavailable."""

    def factory():
        return _make_shadow(detector_name, schema)

    return factory


def make_spoof_env(
    schema_name: str = "falcon9",
    detector_name: str = "base",
    holdout: bool = False,
    config: SpoofEnvConfig | None = None,
) -> SpoofEnv:
    schema = SCHEMAS_BY_NAME[schema_name]
    trajectories = load_training_trajectories(schema_name, holdout=holdout)
    return SpoofEnv(trajectories, make_detector_factory(detector_name, schema), schema, config=config)

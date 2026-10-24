"""Train the PPO attacker against a detector and save the policy + a metrics JSON.

    uv run python scripts/train_ppo.py --schema falcon9 --detector base --timesteps 200000

The real policy (``models/ppo_<schema>.zip``) is only written when the real detector imports. If the
detector module isn't ready, this still runs against the built-in stub to validate the whole pipeline,
but saves to ``models/ppo_<schema>_stub.zip`` and marks the metrics ``detector_available: false`` so a
stub policy is never mistaken for a trained one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vbw.schema import SCHEMAS_BY_NAME
from vbw.attacker.env import (
    SpoofEnvConfig,
    load_training_trajectories,
    make_detector_factory,
    SpoofEnv,
)
from vbw.attacker.evaluate import aggregate, evaluate_attack
from vbw.attacker.registry import make_attack

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def detector_available(name: str, schema) -> bool:
    try:
        from vbw.detector.registry import make_detector

        make_detector(name, schema)
        return True
    except Exception:
        return False


def make_env_thunk(trajectories, det_factory, schema, config):
    def thunk():
        from stable_baselines3.common.monitor import Monitor

        return Monitor(SpoofEnv(trajectories, det_factory, schema, config=config))

    return thunk


def evaluate_policy(model, schema, det_factory, trajectories, start_frac=0.35, seed=0):
    """Mean induced picture-channel error before first alert + alert rate, for the trained policy."""
    from vbw.attacker.env import PpoAttack

    results = []
    for t, x in trajectories:
        start_t = float(t[int(len(t) * start_frac)])
        atk = PpoAttack(schema=schema, start_t=start_t, intensity=1.0)
        atk._policy = model  # drive the in-memory policy directly
        atk.loaded = True
        results.append(evaluate_attack(t, x, schema, atk, det_factory(), start_t, seed=seed))
    return aggregate(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="falcon9")
    ap.add_argument("--detector", default="base")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--episode-len", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="override output .zip path")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    schema = SCHEMAS_BY_NAME[args.schema]
    available = detector_available(args.detector, schema)
    det_factory = make_detector_factory(args.detector, schema)
    trajectories = load_training_trajectories(args.schema)
    config = SpoofEnvConfig(episode_len=args.episode_len)

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    elif available:
        out_path = MODELS / f"ppo_{args.schema}.zip"
    else:
        out_path = MODELS / f"ppo_{args.schema}_stub.zip"

    print(f"[train] schema={args.schema} detector={args.detector} available={available} "
          f"timesteps={args.timesteps} -> {out_path.name}")

    thunks = [make_env_thunk(trajectories, det_factory, schema, config) for _ in range(args.n_envs)]
    venv = SubprocVecEnv(thunks)
    # log_std_init=-1 (std ~0.37): with std 1.0 the noise on the cross-checked channels trips the
    # detector in ~9 steps every episode, so the policy never sees a surviving trajectory to learn from.
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, n_steps=512, batch_size=512,
                gae_lambda=0.95, gamma=0.995, ent_coef=0.0, device="cpu",
                policy_kwargs=dict(log_std_init=-1.0))
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_s = time.time() - t0
    venv.close()

    model.save(str(out_path))
    ppo_eval = evaluate_policy(model, schema, det_factory, trajectories, seed=args.seed)

    # scripted baselines over the same launches, same start, for an apples-to-apples report
    scripted = {}
    for name in ("none", "bias", "drift", "replay", "freeze", "coordinated"):
        runs = []
        for t, x in trajectories:
            start_t = float(t[int(len(t) * 0.35)])
            atk = make_attack(name, schema, start_t=start_t, intensity=1.0)
            runs.append(evaluate_attack(t, x, schema, atk, det_factory(), start_t))
        scripted[name] = aggregate(runs)

    metrics = {
        "schema": args.schema,
        "detector": args.detector,
        "detector_available": available,
        "is_stub": not available,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "train_seconds": round(train_s, 1),
        "model_path": str(out_path),
        "ppo": ppo_eval,
        "scripted": scripted,
    }
    metrics_path = RESULTS / f"ppo_{args.schema}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] saved {out_path.name} in {train_s:.0f}s; metrics -> {metrics_path.name}")
    print(f"[train] PPO mean_lie_before_alert={ppo_eval['mean_lie_before_alert']:.1f} "
          f"detection_rate={ppo_eval['detection_rate']:.2f}")


if __name__ == "__main__":
    main()

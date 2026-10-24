"""Train the PPO attacker against a detector and save the policy + a metrics JSON.

    uv run python scripts/train_ppo.py --schema falcon9 --detector base --timesteps 300000

Always trains against the real detector (``make_detector_factory`` raises if it can't be built) and
writes ``models/ppo_<schema>.zip``. The report evaluates the policy on the training launches and on
the holdout launches it never saw, next to the scripted attacks over the same holdout runs.
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


def make_env_thunk(trajectories, det_factory, schema, config):
    def thunk():
        from stable_baselines3.common.monitor import Monitor

        return Monitor(SpoofEnv(trajectories, det_factory, schema, config=config))

    return thunk


def evaluate_policy(model, schema, det_factory, trajectories, start_frac=0.35, seed=0, detector_name="base"):
    """Mean induced picture-channel error before first alert + alert rate, for the trained policy.

    The policy's shadow detector is the one it trained against, so its score observations match training."""
    from vbw.attacker.env import PpoAttack

    results = []
    for t, x in trajectories:
        start_t = float(t[int(len(t) * start_frac)])
        atk = PpoAttack(schema=schema, start_t=start_t, intensity=1.0, detector_name=detector_name)
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
    ap.add_argument("--metrics-out", default=None,
                    help="override metrics JSON path (default results/ppo_<schema>_metrics.json)")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    schema = SCHEMAS_BY_NAME[args.schema]
    det_factory = make_detector_factory(args.detector, schema)
    det_factory()  # fail before spawning workers if the detector can't be built
    trajectories = load_training_trajectories(args.schema)
    holdout = load_training_trajectories(args.schema, holdout=True, seeds=(0,))
    config = SpoofEnvConfig(episode_len=args.episode_len)

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else MODELS / f"ppo_{args.schema}.zip"

    print(f"[train] schema={args.schema} detector={args.detector} "
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
    import torch

    torch.set_num_threads(1)  # evaluation predicts one frame at a time; extra threads only add overhead
    ppo_train = evaluate_policy(model, schema, det_factory, trajectories[::5], seed=args.seed,
                                detector_name=args.detector)
    ppo_holdout = evaluate_policy(model, schema, det_factory, holdout, seed=args.seed, detector_name=args.detector)

    # scripted baselines over the same holdout launches, same start, for an apples-to-apples report
    scripted = {}
    for name in ("none", "bias", "drift", "replay", "freeze", "coordinated"):
        runs = []
        for t, x in holdout:
            start_t = float(t[int(len(t) * 0.35)])
            atk = make_attack(name, schema, start_t=start_t, intensity=1.0)
            runs.append(evaluate_attack(t, x, schema, atk, det_factory(), start_t))
        scripted[name] = aggregate(runs)

    metrics = {
        "schema": args.schema,
        "detector": args.detector,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "train_seconds": round(train_s, 1),
        "model_path": out_path.name,
        "ppo_train": ppo_train,
        "ppo_holdout": ppo_holdout,
        "scripted_holdout": scripted,
    }
    metrics_path = Path(args.metrics_out) if args.metrics_out else RESULTS / f"ppo_{args.schema}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] saved {out_path.name} in {train_s:.0f}s; metrics -> {metrics_path.name}")
    for split, ev in (("train", ppo_train), ("holdout", ppo_holdout)):
        print(f"[train] PPO {split}: mean_lie_before_alert={ev['mean_lie_before_alert']:.1f} "
              f"max_lie_before_alert={ev['mean_max_lie_before_alert']:.1f} "
              f"detection_rate={ev['detection_rate']:.2f}")


if __name__ == "__main__":
    main()

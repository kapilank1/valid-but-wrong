"""Run attacks over the training launches and write forged-window datasets + an attack-eval report.

    uv run python scripts/make_forged_dataset.py --schema falcon9

Outputs:
- ``data/cache/forged_<schema>_<attack>.npz`` per attack, in the format defined by
  ``vbw/detector/README.md`` (W = features.L3_WINDOW, consecutive frames, y = last-frame-altered).
  Uses ``vbw.detector.features.save_forged_npz`` when available; otherwise writes the keys directly.
- ``results/attacks_eval.json``: each attack vs the real base detector on the holdout launches
  (detected?, latency, max lie before detection).

Every attack runs over the frames the ground station actually decodes (sensor noise + float32 wire
rounding, via ``vbw.detector.data.clean_runs``), never the noise-free CSVs: the detector's
frozen-channel rule fires on those, and L3 would learn "noise-free = clean" instead of the attacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vbw.schema import SCHEMAS_BY_NAME, ChannelSchema, Frame
from vbw.attacker.evaluate import aggregate, evaluate_attack
from vbw.attacker.registry import make_attack

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

SCRIPTED_FOR = {
    "falcon9": ("none", "bias", "drift", "replay", "freeze", "coordinated"),
    "quad": ("none", "bias", "drift", "replay", "freeze", "coordinated"),
}
INTENSITIES = (0.4, 0.7, 1.0)
STRIDE = 10
START_FRAC = 0.35                  # attack onset for the evaluation report (fixed protocol)
# Training onsets are spread over the flight. With a single onset every forged window came from late flight, and L3
# learned "late flight = forged" (it saturated on clean first-stage entry and landing burns).
TRAIN_START_FRACS = (0.1, 0.4, 0.7)


def _train_start_frac(traj_idx: int, variant: int) -> float:
    """Rotate onsets so every launch gets each onset across its intensity variants."""
    return TRAIN_START_FRACS[(traj_idx + variant) % len(TRAIN_START_FRACS)]


def _window(schema: ChannelSchema) -> int:
    try:
        from vbw.detector.features import l3_window

        return int(l3_window(schema))
    except Exception:
        try:
            from vbw.detector.features import L3_WINDOW  # older contract name

            return int(L3_WINDOW)
        except Exception:
            return 160  # current contract default (16 s at 10 Hz)


def _trajectories(schema: ChannelSchema, split: str = "train"):
    """(scenario, t [T], X [T, C]) noisy, wire-rounded runs, exactly what the detector sees."""
    from vbw.detector.data import clean_runs

    return [(run.scenario, run.t, run.X) for run in clean_runs(schema, split)]


def _apply_over(schema: ChannelSchema, t: np.ndarray, x: np.ndarray, attack) -> np.ndarray:
    """Return reported channel array [T, C] after running the attack frame by frame."""
    attack.reset(0)
    out = np.empty_like(x)
    for i in range(len(t)):
        channels = {c: float(x[i, j]) for j, c in enumerate(schema.channels)}
        rep = attack.apply(Frame("f", schema.schema_id, i & 0xFFFF, float(t[i]), channels))
        out[i] = [rep.channels[c] for c in schema.channels]
    return out


def _windows(x: np.ndarray, t: np.ndarray, active: np.ndarray, w: int, stride: int):
    for s in range(0, len(t) - w + 1, stride):
        yield x[s:s + w], t[s:s + w], int(active[s + w - 1])


def _save_npz(path: Path, X, tt, y, attack, scenario, schema_name, w) -> None:
    try:
        from vbw.detector.features import save_forged_npz

        save_forged_npz(path, X=X, t=tt, y=y, attack=np.array(attack), scenario=np.array(scenario),
                        schema=schema_name, window=w)
        return
    except Exception:
        pass
    np.savez_compressed(
        path, X=X.astype(np.float32), t=tt.astype(np.float64), y=y.astype(np.int8),
        attack=np.array(attack), scenario=np.array(scenario),
        schema=np.array(schema_name), window=np.array(w),
    )


def build_datasets(schema: ChannelSchema, w: int) -> dict:
    trajs = _trajectories(schema)
    summary = {}
    for name in SCRIPTED_FOR[schema.name]:
        X, TT, Y, ATK, SCN = [], [], [], [], []
        for k, (scenario, t, x) in enumerate(trajs):
            intens = (0.0,) if name == "none" else INTENSITIES
            for v, inten in enumerate(intens):
                start_t = 0.0 if name == "none" else float(t[int(len(t) * _train_start_frac(k, v))])
                atk = make_attack(name, schema, start_t=start_t, intensity=inten)
                reported = _apply_over(schema, t, x, atk)
                active = (t >= start_t) & (name != "none")
                for xw, tw, yw in _windows(reported, t, active, w, STRIDE):
                    X.append(xw); TT.append(tw); Y.append(yw)
                    ATK.append(name); SCN.append(scenario)
        X = np.asarray(X, dtype=np.float32)
        path = CACHE / f"forged_{schema.name}_{name}.npz"
        _save_npz(path, X, np.asarray(TT), np.asarray(Y, dtype=np.int8), ATK, SCN, schema.name, w)
        pos = int(np.sum(Y)) if len(Y) else 0
        summary[name] = {"file": path.name, "windows": len(Y), "positives": pos}
        print(f"[forge] {path.name}: {len(Y)} windows ({pos} forged)")
    return summary


def eval_attacks(schema: ChannelSchema, detector_name: str = "base") -> dict:
    """Each attack (plus PPO when a trained policy exists) vs the real detector on the holdout runs."""
    from vbw.detector.registry import make_detector

    names = list(SCRIPTED_FOR[schema.name])
    if (ROOT / "models" / f"ppo_{schema.name}.zip").exists():
        names.append("ppo")
    trajs = _trajectories(schema, "holdout")
    out = {"detector": detector_name, "schema": schema.name, "split": "holdout", "attacks": {}}
    for name in names:
        runs = []
        for _, t, x in trajs:
            start_t = 0.0 if name == "none" else float(t[int(len(t) * START_FRAC)])
            atk = make_attack(name, schema, start_t=start_t, intensity=1.0)
            runs.append(evaluate_attack(t, x, schema, atk, make_detector(detector_name, schema), start_t))
        out["attacks"][name] = aggregate(runs)
        print(f"[eval] {name}: detection_rate={out['attacks'][name]['detection_rate']:.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="falcon9")
    ap.add_argument("--ppo-only", metavar="ATTACK", default=None,
                    help="only build forged_<schema>_<ATTACK>.npz for a learned attack (ppo, ppo_r2); leaves the "
                         "scripted sets and results/attacks_eval.json alone")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(1)  # the PPO policy predicts one frame at a time; extra threads only add overhead
    schema = SCHEMAS_BY_NAME[args.schema]
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    w = _window(schema)

    if args.ppo_only:
        _build_ppo_dataset(schema, w, {}, workers=args.workers, attack=args.ppo_only)
        return

    summary = build_datasets(schema, w)

    # PPO forged set if a trained policy exists
    model = ROOT / "models" / f"ppo_{schema.name}.zip"
    if model.exists():
        _build_ppo_dataset(schema, w, summary, workers=args.workers)

    report = eval_attacks(schema)
    (RESULTS / "attacks_eval.json").write_text(json.dumps(report, indent=2))
    (CACHE / f"forged_{schema.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[forge] wrote results/attacks_eval.json (detector={report['detector']})")


def _ppo_run(job: tuple[str, int, float, str]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray] | None:
    """One launch under a learned attack from one onset: (scenario, t, reported, active), or None without a policy."""
    import torch

    torch.set_num_threads(1)
    schema_name, idx, start_frac, attack = job
    schema = SCHEMAS_BY_NAME[schema_name]
    scenario, t, x = _trajectories(schema)[idx]
    start_t = float(t[int(len(t) * start_frac)])
    atk = make_attack(attack, schema, start_t=start_t, intensity=1.0)
    if not atk.loaded:
        return None
    return scenario, t, _apply_over(schema, t, x, atk), t >= start_t


def _build_ppo_dataset(schema: ChannelSchema, w: int, summary: dict, workers: int = 12, attack: str = "ppo") -> None:
    """data/cache/forged_<schema>_<attack>.npz from a learned attack (ppo, ppo_r2) at every training onset."""
    from concurrent.futures import ProcessPoolExecutor

    n = len(_trajectories(schema))
    jobs = [(schema.name, i, f, attack) for i in range(n) for f in TRAIN_START_FRACS]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(_ppo_run, jobs))
    if any(r is None for r in runs):
        print(f"[forge] {attack}: no trained policy, skipped")
        return
    X, TT, Y, ATK, SCN = [], [], [], [], []
    for scenario, t, reported, active in runs:
        for xw, tw, yw in _windows(reported, t, active, w, STRIDE):
            X.append(xw); TT.append(tw); Y.append(yw); ATK.append(attack); SCN.append(scenario)
    path = CACHE / f"forged_{schema.name}_{attack}.npz"
    _save_npz(path, np.asarray(X, dtype=np.float32), np.asarray(TT),
              np.asarray(Y, dtype=np.int8), ATK, SCN, schema.name, w)
    summary[attack] = {"file": path.name, "windows": len(Y), "positives": int(np.sum(Y))}
    print(f"[forge] {path.name}: {len(Y)} windows ({int(np.sum(Y))} forged)")


if __name__ == "__main__":
    main()

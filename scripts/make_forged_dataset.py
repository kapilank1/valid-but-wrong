"""Run attacks over the training launches and write forged-window datasets + an attack-eval report.

    uv run python scripts/make_forged_dataset.py --schema falcon9

Outputs:
- ``data/cache/forged_<schema>_<attack>.npz`` per attack, in the format defined by
  ``vbw/detector/README.md`` (W = features.L3_WINDOW, consecutive frames, y = last-frame-altered).
  Uses ``vbw.detector.features.save_forged_npz`` when available; otherwise writes the keys directly.
- ``results/attacks_eval.json``: each scripted attack vs the detector (detected?, latency, max lie
  before detection). Run against the real base detector when it imports, else the built-in stub.
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
START_FRAC = 0.35


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


def _trajectories(schema_name: str):
    from vbw.detector.splits import load_falcon9, train_slugs

    return [(slug, *load_falcon9(slug)) for slug in train_slugs()]  # (slug, t[T], X[T,C])


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
    trajs = _trajectories(schema.name)
    summary = {}
    for name in SCRIPTED_FOR[schema.name]:
        X, TT, Y, ATK, SCN = [], [], [], [], []
        for slug, t, x in trajs:
            start_t = 0.0 if name == "none" else float(t[int(len(t) * START_FRAC)])
            intens = (0.0,) if name == "none" else INTENSITIES
            for inten in intens:
                atk = make_attack(name, schema, start_t=start_t, intensity=inten)
                reported = _apply_over(schema, t, x, atk)
                active = (t >= start_t) & (name != "none")
                for xw, tw, yw in _windows(reported, t, active, w, STRIDE):
                    X.append(xw); TT.append(tw); Y.append(yw)
                    ATK.append(name); SCN.append(f"{schema.name}:{slug}")
        X = np.asarray(X, dtype=np.float32)
        path = CACHE / f"forged_{schema.name}_{name}.npz"
        _save_npz(path, X, np.asarray(TT), np.asarray(Y, dtype=np.int8), ATK, SCN, schema.name, w)
        pos = int(np.sum(Y)) if len(Y) else 0
        summary[name] = {"file": path.name, "windows": len(Y), "positives": pos}
        print(f"[forge] {path.name}: {len(Y)} windows ({pos} forged)")
    return summary


def eval_attacks(schema: ChannelSchema) -> dict:
    """Each scripted attack vs the detector. Real base detector if available, else stub."""
    try:
        from vbw.detector.registry import make_detector
        from vbw.detector.splits import holdout_slugs, load_falcon9

        det_factory = lambda: make_detector("base", schema)
        slugs = holdout_slugs()
        detector_name = "base"
    except Exception:
        from vbw.attacker.env import StubDetector
        from vbw.detector.splits import holdout_slugs, load_falcon9

        det_factory = lambda: StubDetector(schema)
        slugs = holdout_slugs()
        detector_name = "stub"

    out = {"detector": detector_name, "schema": schema.name, "attacks": {}}
    for name in SCRIPTED_FOR[schema.name]:
        runs = []
        for slug in slugs:
            t, x = load_falcon9(slug)
            start_t = 0.0 if name == "none" else float(t[int(len(t) * START_FRAC)])
            atk = make_attack(name, schema, start_t=start_t, intensity=1.0)
            runs.append(evaluate_attack(t, x, schema, atk, det_factory(), start_t))
        out["attacks"][name] = aggregate(runs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="falcon9")
    args = ap.parse_args()
    schema = SCHEMAS_BY_NAME[args.schema]
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    w = _window(schema)

    summary = build_datasets(schema, w)

    # PPO forged set if a trained policy exists
    model = ROOT / "models" / f"ppo_{schema.name}.zip"
    if model.exists():
        _build_ppo_dataset(schema, w, summary)

    report = eval_attacks(schema)
    (RESULTS / "attacks_eval.json").write_text(json.dumps(report, indent=2))
    (CACHE / f"forged_{schema.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[forge] wrote results/attacks_eval.json (detector={report['detector']})")


def _build_ppo_dataset(schema: ChannelSchema, w: int, summary: dict) -> None:
    from vbw.attacker.env import PpoAttack

    trajs = _trajectories(schema.name)
    X, TT, Y, ATK, SCN = [], [], [], [], []
    for slug, t, x in trajs:
        start_t = float(t[int(len(t) * START_FRAC)])
        atk = PpoAttack(schema=schema, start_t=start_t, intensity=1.0)
        if not atk.loaded:
            return
        reported = _apply_over(schema, t, x, atk)
        active = t >= start_t
        for xw, tw, yw in _windows(reported, t, active, w, STRIDE):
            X.append(xw); TT.append(tw); Y.append(yw); ATK.append("ppo"); SCN.append(f"{schema.name}:{slug}")
    path = CACHE / f"forged_{schema.name}_ppo.npz"
    _save_npz(path, np.asarray(X, dtype=np.float32), np.asarray(TT),
              np.asarray(Y, dtype=np.int8), ATK, SCN, schema.name, w)
    summary["ppo"] = {"file": path.name, "windows": len(Y), "positives": int(np.sum(Y))}
    print(f"[forge] {path.name}: {len(Y)} windows ({int(np.sum(Y))} forged)")


if __name__ == "__main__":
    main()

"""Train the L3 discriminator P(forged) on attacker-forged windows + clean windows, export ONNX.

    uv run python scripts/train_l3.py --schema falcon9

Reads every data/cache/forged_<schema>_*.npz (format in vbw/detector/README.md; files with "holdout" in the name
are evaluation-only), adds clean noisy windows from the training launches, computes the shared per-frame features
(vbw.detector.features.l3_features: L1 residual ratios, L2 z-scores, diffs) and fits a small 1D CNN.

Attacker windows must carry the same sensor noise as the live link (i.e. be built from Falcon9Source frames);
otherwise the classifier learns "noise-free = forged". If a dataset was built from noise-free CSVs, pass
--add-noise to apply the source's own noise model to every attacker window before featurizing.

Writes models/l3_<schema>.onnx and models/l3_<schema>_meta.json (logit threshold calibrated on clean held-out
validation launches, per-attack validation detection rates).
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from vbw.detector.data import clean_runs, wire_round
from vbw.detector.features import (
    L3_CONTEXT, ForgedSet, feature_names, l3_features, l3_window, load_forged_npz,
)
from vbw.detector.l2 import MODELS_DIR, L2Model
from vbw.detector.l3 import l3_paths
from vbw.detector.nets import export_onnx
from vbw.detector.physics import L1Config
from vbw.detector.train import L3FitConfig, fit_l3_model, logits_of
from vbw.schema import SCHEMAS_BY_NAME, ChannelSchema

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CLEAN_STRIDE = 10
VAL_SCENARIO_EVERY = 5     # every 5th training scenario's windows are L3 validation (by launch, no leakage)


def clean_windows(schema: ChannelSchema, split: str, seeds: tuple[int, ...], stride: int) -> ForgedSet:
    W = l3_window(schema)
    X, T, S = [], [], []
    for run in clean_runs(schema, split, seeds):
        for s in range(0, len(run.t) - W + 1, stride):
            X.append(run.X[s:s + W]); T.append(run.t[s:s + W]); S.append(run.scenario)
    n = len(X)
    return ForgedSet(np.asarray(X, np.float32), np.asarray(T), np.zeros(n, np.int8), np.array(["clean"] * n),
                     np.array(S), schema.name, W)


def add_noise(fs: ForgedSet, schema: ChannelSchema) -> ForgedSet:
    if schema.name != "falcon9":
        return fs
    from vbw.sources.falcon9 import add_sensor_noise

    X = np.empty_like(fs.X)
    for i in range(len(fs)):
        rng = np.random.default_rng([i, zlib.crc32(fs.scenario[i].encode())])
        X[i] = wire_round(add_sensor_noise(fs.X[i].astype(np.float64), rng))
    return ForgedSet(X, fs.t, fs.y, fs.attack, fs.scenario, fs.schema, fs.window)


def load_attacker_sets(schema: ChannelSchema, include_holdout: bool = False) -> ForgedSet | None:
    sets = []
    for path in sorted(CACHE.glob(f"forged_{schema.name}_*.npz")):
        if ("holdout" in path.name) != include_holdout:
            continue
        fs = load_forged_npz(path)
        if fs.window != l3_window(schema):
            print(f"  skip {path.name}: window {fs.window} != {l3_window(schema)}")
            continue
        print(f"  {path.name}: {len(fs)} windows, {int(fs.y.sum())} forged")
        sets.append(fs)
    return ForgedSet.concat(sets) if sets else None


def balance(fs: ForgedSet, per_attack: int, rng: np.random.Generator) -> ForgedSet:
    """Cap each (attack, label) group so one big file doesn't drown the others."""
    keep = []
    for name in np.unique(fs.attack):
        for label in (0, 1):
            idx = np.where((fs.attack == name) & (fs.y == label))[0]
            keep.append(rng.choice(idx, min(per_attack, len(idx)), replace=False) if len(idx) else idx)
    return fs.subset(np.sort(np.concatenate(keep)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", choices=["falcon9", "quad"], default="falcon9")
    ap.add_argument("--per-attack", type=int, default=6000, help="max windows per (attack, label) group")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--add-noise", action="store_true", help="add sensor noise to (noise-free) attacker windows")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)
    schema = SCHEMAS_BY_NAME[args.schema]

    l2 = L2Model.load(schema)
    if l2 is None:
        raise SystemExit(f"train L2 first: models/l2_{schema.name}.onnx is missing")
    l1 = L1Config.load(schema)

    print("loading forged windows")
    forged = load_attacker_sets(schema)
    if forged is None:
        raise SystemExit(f"no data/cache/forged_{schema.name}_*.npz found (see vbw/detector/README.md)")
    forged = balance(forged, args.per_attack, rng)
    if args.add_noise:
        forged = add_noise(forged, schema)
    clean = balance(clean_windows(schema, "train", (0, 1), CLEAN_STRIDE), 4 * args.per_attack, rng)
    data = ForgedSet.concat([forged, clean])
    print(f"featurizing {len(data)} windows ({int(data.y.sum())} forged)")
    F = l3_features(data.X, data.t, schema, l1, l2)

    scen = sorted(set(data.scenario))
    val_scen = set(scen[2::VAL_SCENARIO_EVERY])
    is_val = np.isin(data.scenario, list(val_scen))
    model = fit_l3_model(F[~is_val], data.y[~is_val], F[is_val], data.y[is_val], L3FitConfig(epochs=args.epochs))

    # Threshold on clean *held-out* launches (never seen by L1/L2/L3 training), with streaming-like stride.
    clean_val = clean_windows(schema, "val", (100,), 5)
    lv = logits_of(model, l3_features(clean_val.X, clean_val.t, schema, l1, l2))
    logit_thr = float(np.quantile(lv, L3FitConfig.clean_quantile))
    logit_thr = max(logit_thr, 0.0)  # never alert on windows the model itself calls more-likely-clean

    lval = logits_of(model, F[is_val])
    per_attack = {}
    for name in np.unique(data.attack[is_val]):
        m = is_val & (data.attack == name)
        lv_m, y_m = logits_of(model, F[m]), data.y[m]
        per_attack[str(name)] = {
            "windows": int(m.sum()),
            "forged_detect_rate": float(np.mean(lv_m[y_m == 1] > logit_thr)) if (y_m == 1).any() else None,
            "clean_false_rate": float(np.mean(lv_m[y_m == 0] > logit_thr)) if (y_m == 0).any() else None,
        }
    onnx_path, meta_path = l3_paths(schema)
    export_onnx(model, F[:2], onnx_path, "logit")
    meta = {
        "schema": schema.name, "context": L3_CONTEXT, "window": l3_window(schema), "features": feature_names(schema),
        "logit_threshold": logit_thr, "clean_quantile": L3FitConfig.clean_quantile,
        "clean_val_windows": int(len(lv)), "clean_val_rate_over": float(np.mean(lv > logit_thr)),
        "noise_augmented": bool(args.add_noise), "val_scenarios": sorted(val_scen),
        "train_windows": int((~is_val).sum()), "val_windows": int(is_val.sum()),
        "val_per_attack": per_attack, "attacks": sorted(set(map(str, data.attack))),
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("logit_threshold", "clean_val_rate_over", "val_per_attack")}, indent=1))
    print(f"wrote {onnx_path} and {meta_path}")


if __name__ == "__main__":
    main()

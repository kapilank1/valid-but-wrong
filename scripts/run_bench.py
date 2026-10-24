"""Benchmark the bench predictor + dynamic thresholding on NASA SMAP/MSL (telemanom protocol).

    uv run python scripts/run_bench.py                      # full run -> results/bench_summary.json
    uv run python scripts/run_bench.py --channels P-1,M-6 --out-dir /tmp/smoke
    uv run python scripts/run_bench.py --cache-dir /tmp/vbw_preds   # reuse predictions across runs

    uv run python scripts/run_bench.py --residual --tag residual   # ablation (not the headline)

Writes <out-dir>/bench_summary[_tag].json and <out-dir>/bench_channels[_tag].json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vbw.bench.baseline import BASELINE  # noqa: E402
from vbw.bench.data import DATA_DIR, load_labels, unique_channels  # noqa: E402
from vbw.bench.evaluate import evaluate_row, summarize_by_spacecraft  # noqa: E402
from vbw.bench.model import ModelConfig, count_params  # noqa: E402
from vbw.bench.runner import predict_channels  # noqa: E402
from vbw.bench.threshold import ThresholdConfig  # noqa: E402

NOTES = [
    "Protocol: every row of telemanom's labeled_anomalies.csv is one stream (55 SMAP rows / 69 anomalies, "
    "27 MSL rows / 36 anomalies). P-2 (SMAP) is listed twice with slightly different labels; like telemanom "
    "we score both rows, using one model. T-10 has data files but no label row, so it is not evaluated.",
    "Event-level scoring per the paper: a labeled sequence overlapped by any predicted sequence is a TP; "
    "unmatched labeled sequences are FN; predicted sequences overlapping no label are FP.",
    "Hyperparameters are the paper's defaults (l_s=250, 2x80 LSTM, dropout 0.3, batch 64, <=35 epochs, "
    "early stopping patience 10 / min_delta 3e-4; thresholding batch 70, window 30, smoothing 5%, buffer 100, "
    "p=0.13). Nothing was tuned on the test labels.",
    "Differences from telemanom: PyTorch instead of Keras; best-validation weights restored after early "
    "stopping; no C-2 special case (telemanom hard-codes one for that channel); the last thresholding window "
    "judges all its new points.",
    "Baseline F1 is derived from the paper's reported precision/recall; the paper reports F0.5.",
    "Point-adjusted F1 is point-wise with segment-level adjustment. It is optimistic and not comparable to "
    "the event-level numbers.",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6, help="CPU worker processes (1 torch thread each)")
    ap.add_argument("--mps-workers", type=int, default=1 if torch.backends.mps.is_available() else 0,
                    help="extra worker processes on Apple MPS (default 1 if available)")
    ap.add_argument("--channels", default="", help="comma-separated channel ids (default: all labeled)")
    ap.add_argument("--epochs", type=int, default=ModelConfig.epochs)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual", action="store_true", help="ablation: predict change from last value")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=None, help="optional prediction cache (outside the repo)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. 'residual' -> bench_summary_residual.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    rows = load_labels(args.data_dir / "labeled_anomalies.csv")
    if args.channels:
        wanted = {c.strip() for c in args.channels.split(",") if c.strip()}
        rows = [r for r in rows if r.chan_id in wanted]
    chans = unique_channels(rows)
    mcfg = ModelConfig(epochs=args.epochs, seed=args.seed, residual=args.residual)
    tcfg = ThresholdConfig(l_s=mcfg.l_s)
    devices = ["mps"] * args.mps_workers + ["cpu"] * args.workers
    print(f"{len(rows)} label rows, {len(chans)} channels, workers={devices}", flush=True)

    preds = predict_channels(chans, mcfg, devices, args.data_dir, args.cache_dir,
                             log=lambda m: print(m, flush=True))
    t_pred = time.perf_counter() - t0
    evaluated = [evaluate_row(r, preds[r.chan_id], tcfg) for r in rows]
    datasets = summarize_by_spacecraft(evaluated)
    runtime = time.perf_counter() - t0

    train_s = [p.train_s for p in preds.values()]
    summary = {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "name": ("Per-channel LSTM next-step predictor" + (" (residual ablation)" if mcfg.residual else "")
                     + " + nonparametric dynamic thresholding"),
            "framework": f"torch {torch.__version__}",
            "config": mcfg.to_dict(),
            "threshold": tcfg.to_dict(),
            "params_smap": count_params(25, mcfg),
            "params_msl": count_params(55, mcfg),
            "devices": devices,
            "host": f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores",
        },
        "runtime_s": round(runtime, 1),
        "runtime_detail": {
            "train_predict_wall_s": round(t_pred, 1),
            "threshold_eval_s": round(runtime - t_pred, 1),
            "per_channel_train_s_sum": round(sum(train_s), 1),
            "mean_epochs": round(sum(p.epochs_run for p in preds.values()) / max(len(preds), 1), 1),
            "note": "wall times cover only channels not loaded from --cache-dir; train_s sums are from the "
                    "run that produced each prediction",
        },
        "subset": sorted(chans) if args.channels else None,
        "datasets": datasets,
        "baseline": BASELINE,
        "notes": NOTES,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out_summary = args.out_dir / f"bench_summary{suffix}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / f"bench_channels{suffix}.json").write_text(json.dumps(evaluated, indent=1) + "\n")

    for name, d in datasets.items():
        b = BASELINE[name]
        print(f"{name:8s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
              f"(tp={d['tp']} fp={d['fp']} fn={d['fn']}, {d['channels']} rows) | "
              f"telemanom P={b['precision']:.3f} R={b['recall']:.3f} F1*={b['f1']:.3f} | "
              f"PA-F1={d['point_adjusted_optimistic']['f1']:.3f}")
    print(f"runtime {runtime:.0f}s -> {out_summary}")


if __name__ == "__main__":
    main()

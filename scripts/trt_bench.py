"""Build TensorRT FP32 / FP16 / INT8 engines for the detector models and benchmark them against ONNX Runtime (CPU).

    uv run --with tensorrt-cu12 --with "cuda-python>=12.6,<13" --with onnxconverter-common python scripts/trt_bench.py

For L2 (GRU next-step predictor) and L3 (forgery discriminator):
- latency: one window per call (the streaming case: one vehicle, one frame), copies in and out included;
- throughput: big batches (many vehicle streams scored together), and how many streams that sustains at 50 Hz with
  each batch done inside the 20 ms frame period;
- accuracy vs the FP32 ONNX Runtime reference on held-out flights: L2 z-score error and how many frames flip the
  L2 spike decision (|z| over its threshold); L3 logit error and how many windows flip the forged/clean decision.
Engines (GPU-specific) go to models/trt/*.engine; the FP16 / INT8 ONNX graphs next to them. Writes
results/trt_bench.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vbw.detector.data import clean_runs
from vbw.detector.features import l3_features, l3_window, load_forged_npz
from vbw.detector.l2 import L2Model, training_pairs
from vbw.detector.l3 import L3Model
from vbw.detector.physics import L1Config
from vbw.detector.trt import PRECISIONS, TrtModel, build_engine, to_fp16, to_int8
from vbw.schema import FALCON9

ROOT = Path(__file__).resolve().parents[1]
MODELS, CACHE, RESULTS = ROOT / "models", ROOT / "data" / "cache", ROOT / "results"
TRT_DIR = MODELS / "trt"
BATCHES = (64, 256, 1024, 4096, 16384)   # TensorRT has no GRU tactic for a 65,536 batch on this GPU
FRAME_BUDGET_S = 1 / 50


def _percentiles(times: list[float]) -> dict:
    a = np.asarray(times) * 1e6
    return {"p50_us": round(float(np.percentile(a, 50)), 1), "p99_us": round(float(np.percentile(a, 99)), 1)}


def time_calls(fn, x: np.ndarray, iters: int, warmup: int = 50) -> list[float]:
    for _ in range(warmup):
        fn(x)
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        out.append(time.perf_counter() - t0)
    return out


def ort_session(path: Path, threads: int | None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {"x": x})[0]


def speed(fn_single, fn_batch, data: np.ndarray, iters: int) -> dict:
    single = time_calls(fn_single, data[:1], iters)
    out = {"batch1": _percentiles(single), "batches": {}}
    best_streams = 0
    for b in BATCHES:
        x = np.resize(data, (b, *data.shape[1:])).astype(np.float32)
        t = time_calls(fn_batch, x, max(20, iters // max(1, b // 64)), warmup=5)
        p50 = float(np.median(t))
        out["batches"][str(b)] = {"p50_ms": round(p50 * 1e3, 3), "windows_per_s": round(b / p50)}
        if p50 <= FRAME_BUDGET_S:
            best_streams = max(best_streams, b)
    out["streams_at_50hz"] = best_streams   # largest tested batch that finishes inside one 20 ms frame
    return out


# --------------------------------------------------------------------------- data


def l2_data(l2: L2Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(calibration inputs from training flights, held-out inputs, held-out targets)."""
    rng = np.random.default_rng(0)
    cal = np.concatenate([training_pairs(r.X, l2.norm, stride=7)[0] for r in clean_runs(FALCON9, "train", (0,))])
    cal = cal[rng.choice(len(cal), min(4000, len(cal)), replace=False)]
    pairs = [training_pairs(r.X, l2.norm) for r in clean_runs(FALCON9, "holdout", (0,))]
    return cal, np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def l3_data(l1: L1Config, l2: L2Model) -> tuple[np.ndarray, np.ndarray]:
    """(calibration features, evaluation features): forged + clean windows; eval = held-out clean + attack windows."""
    rng = np.random.default_rng(0)
    W = l3_window(FALCON9)
    sets = [load_forged_npz(CACHE / f"forged_falcon9_{a}.npz") for a in ("ppo", "drift", "coordinated", "replay")
            if (CACHE / f"forged_falcon9_{a}.npz").exists()]
    X = np.concatenate([s.X for s in sets]); T = np.concatenate([s.t for s in sets])
    pick = rng.choice(len(X), min(6000, len(X)), replace=False)
    cal_idx, eval_idx = pick[:3000], pick[3000:]
    clean_X, clean_T = [], []
    for run in clean_runs(FALCON9, "holdout", (0,)):
        for s in range(0, len(run.t) - W + 1, 5):
            clean_X.append(run.X[s:s + W]); clean_T.append(run.t[s:s + W])
    cal = l3_features(X[cal_idx], T[cal_idx], FALCON9, l1, l2)
    ev = np.concatenate([l3_features(X[eval_idx], T[eval_idx], FALCON9, l1, l2),
                         l3_features(np.asarray(clean_X), np.asarray(clean_T), FALCON9, l1, l2)])
    return cal.astype(np.float32), ev.astype(np.float32)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--int8-calib", choices=["minmax", "percentile", "entropy"], default="minmax")
    args = ap.parse_args()
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    l1, l2 = L1Config.load(FALCON9), L2Model.load(FALCON9)
    l3 = L3Model.load(FALCON9)

    print("preparing data")
    l2_cal, l2_eval, l2_tgt = l2_data(l2)
    l3_cal, l3_eval = l3_data(l1, l2)
    report: dict = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gpu": None,
                    "cpu": platform.processor(), "models": {}}
    from cuda.bindings import runtime as rt
    import tensorrt as trt

    props = rt.cudaGetDeviceProperties(0)[1]
    report["gpu"] = {"name": props.name.decode(), "compute_capability": f"{props.major}.{props.minor}"}
    report["tensorrt"] = trt.__version__
    report["int8_calibration"] = args.int8_calib

    for name, src, cal, ev, shape in (("l2", MODELS / "l2_falcon9.onnx", l2_cal, l2_eval, l2_eval.shape[1:]),
                                      ("l3", MODELS / "l3_falcon9.onnx", l3_cal, l3_eval, l3_eval.shape[1:])):
        print(f"== {name}: {src.name}, eval windows {len(ev)}")
        ref_fn = ort_session(src, None)
        ref = np.concatenate([ref_fn(ev[i:i + 4096]) for i in range(0, len(ev), 4096)])
        entry = {"input_shape": list(shape), "eval_windows": int(len(ev)),
                 "onnxruntime_cpu": speed(ort_session(src, 1), ort_session(src, None), ev, args.iters),
                 "tensorrt": {}}
        for prec in PRECISIONS:
            onnx_path = {"fp32": src, "fp16": TRT_DIR / f"{name}_falcon9_fp16.onnx",
                         "int8": TRT_DIR / f"{name}_falcon9_int8.onnx"}[prec]
            if prec == "fp16":
                to_fp16(src, onnx_path)
            elif prec == "int8":
                to_int8(src, onnx_path, cal, method=args.int8_calib)
            engine_path = TRT_DIR / f"{name}_falcon9_{prec}.engine"
            t0 = time.time()
            layers = build_engine(onnx_path, engine_path, "x", tuple(shape), max_batch=max(BATCHES))
            build_s = time.time() - t0
            model = TrtModel(engine_path)
            out = np.concatenate([model(ev[i:i + 4096]) for i in range(0, len(ev), 4096)])
            acc = accuracy(name, out, ref, l2, l2_tgt, l3)
            res = {"build_s": round(build_s, 1), "layers": layers, "engine_kb": engine_path.stat().st_size // 1024,
                   "accuracy": acc, **speed(model, model, ev, args.iters)}
            entry["tensorrt"][prec] = res
            model.close()
            print(f"   {prec}: batch1 p50 {res['batch1']['p50_us']} us, 50 Hz streams {res['streams_at_50hz']}, "
                  f"layers {layers}, acc {acc}")
        cpu = entry["onnxruntime_cpu"]
        print(f"   ORT CPU: batch1 p50 {cpu['batch1']['p50_us']} us, 50 Hz streams {cpu['streams_at_50hz']}")
        report["models"][name] = entry

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "trt_bench.json").write_text(json.dumps(report, indent=1))
    print("wrote results/trt_bench.json")


def accuracy(name: str, out: np.ndarray, ref: np.ndarray, l2: L2Model, l2_tgt: np.ndarray, l3: L3Model | None) -> dict:
    if name == "l2":
        z_ref = l2.residuals_to_z(l2_tgt - ref)
        z_out = l2.residuals_to_z(l2_tgt - out)
        thr = l2.norm.inst_threshold
        flips = np.mean((np.abs(z_ref).max(1) > thr) != (np.abs(z_out).max(1) > thr))
        dz = np.abs(z_out - z_ref)
        return {"max_abs_z_err": round(float(dz.max()), 4), "p99_abs_z_err": round(float(np.percentile(dz, 99)), 4),
                "spike_decision_flips": float(flips), "frames": int(len(out))}
    thr = l3.logit_threshold if l3 is not None else 0.0
    d = np.abs(out - ref)
    flips = np.mean((ref > thr) != (out > thr))
    return {"max_abs_logit_err": round(float(d.max()), 4), "p99_abs_logit_err": round(float(np.percentile(d, 99)), 4),
            "decision_flips": float(flips), "windows": int(len(out)), "logit_threshold": round(float(thr), 3)}


if __name__ == "__main__":
    main()

"""Train/predict every channel in parallel worker processes (one torch thread each, optional MPS worker).

Single-threaded CPU LSTM training scales almost linearly across processes on Apple Silicon, while one
MPS worker is ~5x a single CPU thread at batch 64, so the default mixes both.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Callable

import numpy as np

from vbw.bench.data import DATA_DIR, load_array, load_channel
from vbw.bench.model import ChannelPrediction, ModelConfig, fit_and_predict


def config_key(cfg: ModelConfig) -> str:
    return hashlib.sha1(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _cache_paths(cache_dir: Path, chan_id: str, cfg: ModelConfig) -> tuple[Path, Path]:
    base = cache_dir / config_key(cfg)
    return base / f"{chan_id}.npz", base / f"{chan_id}.json"


def load_cached(cache_dir: Path, chan_id: str, cfg: ModelConfig) -> ChannelPrediction | None:
    arr_path, meta_path = _cache_paths(cache_dir, chan_id, cfg)
    if not (arr_path.exists() and meta_path.exists()):
        return None
    with np.load(arr_path, allow_pickle=False) as z:
        y_hat, y_true = z["y_hat"], z["y_true"]
    meta = json.loads(meta_path.read_text())
    return ChannelPrediction(chan_id, y_hat, y_true, meta["epochs_run"], meta["best_val_loss"],
                             meta["n_train_windows"], meta["train_s"], meta["device"])


def save_cached(cache_dir: Path, pred: ChannelPrediction, cfg: ModelConfig) -> None:
    arr_path, meta_path = _cache_paths(cache_dir, pred.chan_id, cfg)
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arr_path, y_hat=pred.y_hat, y_true=pred.y_true)
    meta = {k: getattr(pred, k) for k in ("epochs_run", "best_val_loss", "n_train_windows", "train_s", "device")}
    meta_path.write_text(json.dumps(meta))


def _worker(device: str, tasks: mp.Queue, results: mp.Queue, cfg_dict: dict, data_dir: str) -> None:
    import torch

    torch.set_num_threads(1)
    cfg = ModelConfig(**cfg_dict)
    while True:
        chan_id = tasks.get()
        if chan_id is None:
            return
        try:
            train, test = load_channel(chan_id, Path(data_dir))
            results.put(("ok", chan_id, fit_and_predict(chan_id, train, test, cfg, device)))
        except Exception as exc:  # report and keep serving other channels
            results.put(("error", chan_id, f"{type(exc).__name__}: {exc}"))


def train_size(chan_id: str, data_dir: Path) -> int:
    return int(load_array(data_dir / "train" / f"{chan_id}.npy").shape[0])


def predict_channels(chan_ids: list[str], cfg: ModelConfig, devices: list[str], data_dir: Path = DATA_DIR,
                     cache_dir: Path | None = None, log: Callable[[str], None] = print) -> dict[str, ChannelPrediction]:
    """Return predictions for every channel, using cached ones when available."""
    out: dict[str, ChannelPrediction] = {}
    todo = []
    for chan_id in chan_ids:
        cached = load_cached(cache_dir, chan_id, cfg) if cache_dir else None
        if cached is not None:
            out[chan_id] = cached
        else:
            todo.append(chan_id)
    if out:
        log(f"loaded {len(out)} cached channel predictions")
    if not todo:
        return out

    todo.sort(key=lambda c: -train_size(c, data_dir))  # longest first for load balancing
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for chan_id in todo:
        tasks.put(chan_id)
    devices = devices[: len(todo)] or ["cpu"]
    for _ in devices:
        tasks.put(None)
    procs = [ctx.Process(target=_worker, args=(d, tasks, results, cfg.to_dict(), str(data_dir)), daemon=True)
             for d in devices]
    for p in procs:
        p.start()

    t0, errors = time.perf_counter(), []
    pending = set(todo)
    while pending:
        try:
            status, chan_id, payload = results.get(timeout=30)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                raise RuntimeError(f"all workers exited with {len(pending)} channels pending: {sorted(pending)}")
            continue
        pending.discard(chan_id)
        if status == "ok":
            out[chan_id] = payload
            if cache_dir:
                save_cached(cache_dir, payload, cfg)
            log(f"[{len(todo) - len(pending):>2}/{len(todo)}] {chan_id:5s} {payload.device:3s} "
                f"epochs={payload.epochs_run:>2} val={payload.best_val_loss:.5f} {payload.train_s:6.1f}s "
                f"(elapsed {time.perf_counter() - t0:6.0f}s)")
        else:
            errors.append(f"{chan_id}: {payload}")
            log(f"ERROR {chan_id}: {payload}")
    for p in procs:
        p.join(timeout=10)
    if errors:
        raise RuntimeError("channel failures: " + "; ".join(errors))
    return out

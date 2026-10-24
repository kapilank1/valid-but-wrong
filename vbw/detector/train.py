"""Training + calibration for the learned layers (L2 predictor, L3 discriminator). Scripts in scripts/ are thin CLIs.

Torch is only imported here and in nets.py; the runtime detector needs just onnxruntime.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter
from torch import nn

from vbw.detector.data import Run
from vbw.detector.l2 import L2_WINDOW, L2Model, L2Norm, model_paths, training_pairs
from vbw.detector.nets import GRUPredictor, export_onnx
from vbw.schema import ChannelSchema


MAD_FLOOR = 0.02   # normalized delta units; well below the sensor-noise residual of every channel


@dataclass
class FitConfig:
    epochs: int = 6
    batch: int = 1024
    lr: float = 2e-3
    hidden: int = 64
    stride: int = 1
    max_pairs: int | None = None      # subsample for quick runs / tests
    seed: int = 0
    log: bool = True


def fit_norm(schema: ChannelSchema, runs: list[Run], window: int = L2_WINDOW) -> L2Norm:
    X = np.concatenate([r.X for r in runs])
    D = np.concatenate([np.diff(r.X, axis=0) for r in runs])
    C = len(schema.channels)
    return L2Norm(
        channels=list(schema.channels), window=window,
        mean=X.mean(0).tolist(), std=np.maximum(X.std(0), 1e-6).tolist(), dstd=np.maximum(D.std(0), 1e-6).tolist(),
        res_median=[0.0] * C, res_mad=[1.0] * C,
    )


def _pairs(runs: list[Run], norm: L2Norm, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = zip(*(training_pairs(r.X, norm, stride) for r in runs))
    return np.concatenate(xs), np.concatenate(ys)


def _batches(n: int, batch: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, batch):
        yield order[i: i + batch]


def fit_l2_model(norm: L2Norm, train: list[Run], val: list[Run], cfg: FitConfig) -> GRUPredictor:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Xtr, Ytr = _pairs(train, norm, cfg.stride)
    if cfg.max_pairs and len(Xtr) > cfg.max_pairs:
        keep = rng.choice(len(Xtr), cfg.max_pairs, replace=False)
        Xtr, Ytr = Xtr[keep], Ytr[keep]
    Xva, Yva = _pairs(val, norm, max(cfg.stride, 5))
    model = GRUPredictor(Xtr.shape[-1], Ytr.shape[-1], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    steps = cfg.epochs * math.ceil(len(Xtr) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(steps, 1))
    loss_fn = nn.HuberLoss(delta=1.0)  # OCR glitches in the real data shouldn't dominate the fit
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for idx in _batches(len(Xtr), cfg.batch, rng):
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([float(loss_fn(model(xv[i:i + 8192]), yv[i:i + 8192])) for i in range(0, len(xv), 8192)]))
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
        if cfg.log:
            print(f"  epoch {epoch + 1}/{cfg.epochs} train {total / len(Xtr):.4f} val {val_loss:.4f} ({time.time() - t0:.0f}s, {len(Xtr)} pairs)")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def ewma_series(z: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA along axis 0 starting from 0, identical to the streaming update."""
    return lfilter([alpha], [1.0, -(1.0 - alpha)], z, axis=0)


def calibrate_l2(model: L2Model, val: list[Run], margin: float = 1.25) -> dict:
    """Set residual median/MAD and the instant / EWMA aggregate thresholds on clean validation runs."""
    norm = model.norm
    start = model.needs - 1
    pairs = [model.residual_series(r.X) for r in val]
    res = np.concatenate([p[0][start:] for p in pairs])
    tgt = np.concatenate([p[1][start:] for p in pairs])
    C = len(norm.channels)
    med, mad = np.zeros(C), np.ones(C)
    for c in range(C):
        # Frames where the channel genuinely doesn't move (q = 0 in vacuum) would collapse the MAD to ~0.
        live = res[tgt[:, c] != 0, c]
        if live.size:
            med[c] = np.median(live)
            mad[c] = max(1.4826 * np.median(np.abs(live - med[c])), MAD_FLOOR)
    norm.res_median, norm.res_mad = med.tolist(), mad.tolist()
    model._med, model._mad = med, mad

    inst, slow = [], []
    for r in val:
        z = model.zscore_series(r.X)[start:]
        inst.append(np.abs(z).max(1))
        slow.append(np.abs(ewma_series(z, norm.ewma_alpha)).max(1))
    inst_all, slow_all = np.concatenate(inst), np.concatenate(slow)
    stats = {
        "inst_p99": float(np.percentile(inst_all, 99)), "inst_p99_99": float(np.percentile(inst_all, 99.99)),
        "inst_max": float(inst_all.max()),
        "ewma_p99": float(np.percentile(slow_all, 99)), "ewma_p99_99": float(np.percentile(slow_all, 99.99)),
        "ewma_max": float(slow_all.max()),
    }
    # p99.99 not max: the real data has a handful of OCR glitch frames; the alert layer's persistence absorbs them.
    norm.inst_threshold = margin * stats["inst_p99_99"]
    norm.ewma_threshold = margin * stats["ewma_p99_99"]
    return stats


def train_and_export_l2(schema: ChannelSchema, train: list[Run], val: list[Run], cfg: FitConfig,
                        models_dir: Path) -> tuple[L2Model, dict]:
    norm = fit_norm(schema, train)
    model = fit_l2_model(norm, train, val, cfg)
    onnx_path, norm_path = model_paths(schema, models_dir)
    example = training_pairs(train[0].X[: L2_WINDOW + 10], norm)[0][:2]
    export_onnx(model, example, onnx_path, "delta")
    l2 = L2Model(onnx_path, norm)
    stats = calibrate_l2(l2, val)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(example)).numpy()
    parity = float(np.max(np.abs(torch_out - l2.predict(example))))
    norm.meta = {"train_scenarios": sorted({r.scenario for r in train}), "val_scenarios": sorted({r.scenario for r in val}),
                 "hidden": cfg.hidden, "epochs": cfg.epochs, "onnx_torch_max_abs_diff": parity, "val_stats": stats}
    norm.save(norm_path)
    return l2, stats


# --------------------------------------------------------------------------- L3 discriminator


@dataclass
class L3FitConfig:
    epochs: int = 8
    batch: int = 512
    lr: float = 2e-3
    hidden: int = 32
    seed: int = 0
    clean_quantile: float = 0.9995   # logit threshold = this quantile of clean validation windows
    log: bool = True


def fit_l3_model(F_train: np.ndarray, y_train: np.ndarray, F_val: np.ndarray, y_val: np.ndarray,
                 cfg: L3FitConfig) -> "torch.nn.Module":
    from vbw.detector.nets import WindowClassifier

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = WindowClassifier(F_train.shape[-1], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    pos = max(float(y_train.mean()), 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))  # balance clean vs forged
    xt, yt = torch.from_numpy(F_train), torch.from_numpy(y_train.astype(np.float32))
    xv, yv = torch.from_numpy(F_val), torch.from_numpy(y_val.astype(np.float32))
    best, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for idx in _batches(len(xt), cfg.batch, rng):
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv)) if len(xv) else total / len(xt)
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
        if cfg.log:
            print(f"  epoch {epoch + 1}/{cfg.epochs} train {total / len(xt):.4f} val {val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def logits_of(model: "torch.nn.Module", F: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(F[i:i + 8192])).numpy() for i in range(0, len(F), 8192)]) \
            if len(F) else np.zeros(0)

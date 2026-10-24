"""Per-channel next-step predictor, following telemanom (Hundman et al. 2018) in spirit.

Inputs are the previous `l_s` timesteps of every column (telemetry + command one-hots); outputs are the
next `n_predictions` telemetry values. Like telemanom we train on all horizons and score only the first
(one-step-ahead) prediction. Defaults mirror the paper's Model Parameters table: 2x80 LSTM, dropout 0.3,
l_s=250, batch 64, Adam, up to 35 epochs with early stopping on a 20% validation split.

`residual=True` is an ablation, not telemanom: the last observed value is added to the output, so the
network learns the change rather than the level.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    l_s: int = 250                 # input window (timesteps)
    n_predictions: int = 10        # horizons trained; only the first is scored
    hidden: int = 80
    layers: int = 2
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 35
    patience: int = 10             # Keras-style early stopping on val loss
    min_delta: float = 3e-4
    val_split: float = 0.2
    lr: float = 1e-3
    predict_batch: int = 1024
    seed: int = 0
    residual: bool = False         # ablation: predict the change from the last observed value

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChannelPrediction:
    """One-step-ahead predictions for a channel's test set. Index i corresponds to test index i + l_s."""

    chan_id: str
    y_hat: np.ndarray
    y_true: np.ndarray
    epochs_run: int
    best_val_loss: float
    n_train_windows: int
    train_s: float
    device: str


class Predictor(nn.Module):
    """Stacked LSTM with dropout after each layer (as in telemanom's Keras model) and a linear head."""

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.rnn = nn.LSTM(n_features, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                           dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, cfg.n_predictions)
        self.residual = cfg.residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        delta = self.head(self.drop(out[:, -1]))
        return delta + x[:, -1:, 0] if self.residual else delta


def n_windows(length: int, cfg: ModelConfig) -> int:
    """Number of (input, target) windows telemanom carves from a series of `length` steps."""
    return max(length - cfg.l_s - cfg.n_predictions, 0)


def window_view(arr: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """Zero-copy view of shape [n_windows, l_s + n_predictions, features]."""
    span = cfg.l_s + cfg.n_predictions
    view = sliding_window_view(arr, span, axis=0)  # [n+1, features, span]
    return view[: n_windows(len(arr), cfg)].transpose(0, 2, 1)


def gather_batch(view: np.ndarray, idx: np.ndarray, cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Materialise (X, y) for the given window indices. X: [B, l_s, F], y: [B, n_predictions]."""
    w = np.ascontiguousarray(view[idx])
    return w[:, : cfg.l_s, :], w[:, cfg.l_s :, 0]


def channel_seed(chan_id: str, base: int) -> int:
    return (zlib.crc32(chan_id.encode()) + base) % (2**31)


def _to(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x).to(device, non_blocking=True)


@torch.no_grad()
def _eval_loss(model: Predictor, view: np.ndarray, idx: np.ndarray, cfg: ModelConfig, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(idx), cfg.predict_batch):
        x, y = gather_batch(view, idx[s : s + cfg.predict_batch], cfg)
        pred = model(_to(x, device))
        total += float(((pred - _to(y, device)) ** 2).mean(dim=1).sum())
        count += len(x)
    return total / max(count, 1)


def train_model(train: np.ndarray, cfg: ModelConfig, device: str, seed: int) -> tuple[Predictor, int, float, int]:
    """Fit a predictor on one channel's training data. Returns (model, epochs_run, best_val_loss, n_windows)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    view = window_view(train, cfg)
    n = len(view)
    if n < 2:
        raise ValueError(f"training series too short ({len(train)} steps) for l_s={cfg.l_s}")
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * cfg.val_split)))
    val_idx, fit_idx = np.sort(perm[:n_val]), perm[n_val:]

    model = Predictor(train.shape[1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_state, wait, epochs_run = float("inf"), None, 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(fit_idx)
        for s in range(0, len(order), cfg.batch_size):
            x, y = gather_batch(view, order[s : s + cfg.batch_size], cfg)
            opt.zero_grad(set_to_none=True)
            loss = ((model(_to(x, device)) - _to(y, device)) ** 2).mean()
            loss.backward()
            opt.step()
        epochs_run = epoch + 1
        val = _eval_loss(model, view, val_idx, cfg, device)
        if val < best - cfg.min_delta:  # Keras EarlyStopping semantics; best weights are restored below
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epochs_run, best, n


@torch.no_grad()
def predict_next(model: Predictor, test: np.ndarray, cfg: ModelConfig, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead predictions over the test set. Returns (y_hat, y_true), both of length n_windows."""
    model.eval()
    view = window_view(test, cfg)
    idx = np.arange(len(view))
    preds, trues = [], []
    for s in range(0, len(idx), cfg.predict_batch):
        x, y = gather_batch(view, idx[s : s + cfg.predict_batch], cfg)
        preds.append(model(_to(x, device))[:, 0].float().cpu().numpy())
        trues.append(y[:, 0])
    return np.concatenate(preds), np.concatenate(trues)


def fit_and_predict(chan_id: str, train: np.ndarray, test: np.ndarray, cfg: ModelConfig, device: str) -> ChannelPrediction:
    """Train on `train`, predict `test`. Pure apart from torch RNG seeded from the channel id."""
    t0 = time.perf_counter()
    model, epochs_run, best_val, n_train = train_model(train, cfg, device, channel_seed(chan_id, cfg.seed))
    y_hat, y_true = predict_next(model, test, cfg, device)
    return ChannelPrediction(chan_id, y_hat.astype(np.float64), y_true.astype(np.float64), epochs_run,
                             float(best_val), n_train, time.perf_counter() - t0, device)


def count_params(n_features: int, cfg: ModelConfig) -> int:
    return sum(p.numel() for p in Predictor(n_features, cfg).parameters())

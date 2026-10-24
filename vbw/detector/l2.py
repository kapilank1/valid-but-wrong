"""L2 learned next-step predictor: a small GRU (exported to ONNX) and its robust residual z-scores.

Frame k is judged by predicting its delta (x_k - x_{k-1}) from the previous L2_WINDOW frames. The prediction error,
per channel, is turned into a robust z-score with the median / MAD of errors on clean validation launches. The
detector then aggregates z two ways (calibrated thresholds in the norm file):

* instant: max |z| over channels on this frame (spikes, steps),
* ewma: max |EWMA(z)| over channels (a small but persistent bias in the residual, e.g. a slow drift).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from vbw.schema import ChannelSchema

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
L2_WINDOW = 20
Z_CLIP = 50.0


def model_paths(schema: ChannelSchema, models_dir: Path = MODELS_DIR) -> tuple[Path, Path]:
    return models_dir / f"l2_{schema.name}.onnx", models_dir / f"l2_{schema.name}_norm.json"


@dataclass
class L2Norm:
    """Per-channel normalization and residual calibration, saved next to the ONNX model."""

    channels: list[str]
    window: int
    mean: list[float]
    std: list[float]
    dstd: list[float]                 # std of per-frame deltas
    res_median: list[float]           # clean residual median per channel (normalized units)
    res_mad: list[float]              # clean residual MAD per channel, already × 1.4826
    inst_threshold: float = 8.0       # max|z| that counts as ratio 1
    ewma_alpha: float = 0.1
    ewma_threshold: float = 3.0       # max|EWMA z| that counts as ratio 1
    meta: dict | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "L2Norm":
        return cls(**json.loads(path.read_text()))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.mean), np.asarray(self.std), np.asarray(self.dstd)


def l2_inputs(X: np.ndarray, norm: L2Norm) -> np.ndarray:
    """[..., W+1, C] raw frames -> [..., W, 2C] float32 model input (normalized values of the last W, and deltas)."""
    mean, std, dstd = norm.arrays()
    values = (X[..., 1:, :] - mean) / std
    deltas = np.diff(X, axis=-2) / dstd
    return np.concatenate([values, deltas], axis=-1).astype(np.float32)


def l2_target(X_prev: np.ndarray, X_next: np.ndarray, norm: L2Norm) -> np.ndarray:
    """Normalized delta the model should predict."""
    return ((X_next - X_prev) / np.asarray(norm.dstd)).astype(np.float32)


def training_pairs(X: np.ndarray, norm: L2Norm, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """All (input, target) pairs of one [T, C] run."""
    W = norm.window
    span = sliding_window_view(X, W + 2, axis=0).transpose(0, 2, 1)[::stride]   # [N, W+2, C]
    return l2_inputs(span[:, :-1], norm), l2_target(span[:, -2], span[:, -1], norm)


class L2Model:
    """ONNX Runtime inference plus z-scoring. Loaded once per detector; stateless."""

    def __init__(self, onnx_path: Path, norm: L2Norm):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # streaming batch of 1: threads only add overhead
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.norm = norm
        self._med = np.asarray(norm.res_median)
        self._mad = np.maximum(np.asarray(norm.res_mad), 1e-6)

    @classmethod
    def load(cls, schema: ChannelSchema, models_dir: Path = MODELS_DIR) -> "L2Model | None":
        onnx_path, norm_path = model_paths(schema, models_dir)
        if not (onnx_path.exists() and norm_path.exists()):
            return None
        return cls(onnx_path, L2Norm.load(norm_path))

    @property
    def needs(self) -> int:
        """Frames of history needed to score the newest frame."""
        return self.norm.window + 2

    def predict(self, inputs: np.ndarray, batch: int = 8192) -> np.ndarray:
        outs = [self.session.run(None, {"x": inputs[i: i + batch]})[0] for i in range(0, len(inputs), batch)]
        return np.concatenate(outs) if outs else np.zeros((0, len(self.norm.channels)), np.float32)

    def residuals_to_z(self, residual: np.ndarray) -> np.ndarray:
        return np.clip((residual - self._med) / self._mad, -Z_CLIP, Z_CLIP)

    def zscore_last(self, X: np.ndarray) -> np.ndarray:
        """Signed z per channel of the newest frame of a [>= W+2, C] window."""
        span = X[-self.needs:]
        pred = self.predict(l2_inputs(span[None, :-1], self.norm))[0]
        return self.residuals_to_z(l2_target(span[-2], span[-1], self.norm) - pred)

    def zscore_series(self, X: np.ndarray, last: int | None = None) -> np.ndarray:
        """Signed z per channel for every frame of [..., T, C] (0 where history is too short).

        `last` limits the work to the final `last` frames (the rest are returned as 0).
        """
        lead, (T, C) = X.shape[:-2], X.shape[-2:]
        Xb = X.reshape((-1, T, C))
        z = np.zeros(Xb.shape, dtype=np.float64)
        first = self.needs - 1 if last is None else max(self.needs - 1, T - last)
        if first < T:
            span = sliding_window_view(Xb, self.needs, axis=1).transpose(0, 1, 3, 2)[:, first - self.needs + 1:]
            n, k = span.shape[:2]                                   # [B, K, W+2, C]
            flat = span.reshape((n * k, self.needs, C))
            pred = self.predict(l2_inputs(flat[:, :-1], self.norm))
            res = l2_target(flat[:, -2], flat[:, -1], self.norm) - pred
            z[:, first:] = self.residuals_to_z(res).reshape((n, k, C))
        return z.reshape(lead + (T, C))


def ewma_update(state: np.ndarray, z: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * state + alpha * z


def l2_ratio(z: np.ndarray, ewma: np.ndarray, norm: L2Norm) -> float:
    """Combined L2 ratio: 1.0 means as large as the worst clean validation frame (× margin)."""
    inst = float(np.max(np.abs(z))) / norm.inst_threshold
    slow = float(np.max(np.abs(ewma))) / norm.ewma_threshold
    return max(inst, slow)

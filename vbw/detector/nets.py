"""Torch modules for the learned layers. Only training scripts and tests import this; runtime uses ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GRUPredictor(nn.Module):
    """L2: window of [values, deltas] (normalized) -> normalized delta of the next frame."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class WindowClassifier(nn.Module):
    """L3: per-frame features over a short context -> logit of P(forged). Small dilated 1D CNN + pooling."""

    def __init__(self, n_feat: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))                       # [B, hidden, T]
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], 1)  # [B, 2*hidden]
        return self.head(pooled)[:, 0]


def export_onnx(model: nn.Module, example: np.ndarray, path: Path, output_name: str) -> None:
    """Export with a dynamic batch axis so streaming (B=1) and offline batches share one file."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (torch.from_numpy(example.astype(np.float32)),), str(path),
        input_names=["x"], output_names=[output_name],
        dynamic_axes={"x": {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17, dynamo=False,
    )

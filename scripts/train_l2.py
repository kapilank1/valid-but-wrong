"""Train the L2 GRU next-step predictor on clean training runs and export ONNX + normalization.

    uv run python scripts/train_l2.py --schema falcon9          # ~5 min on an M4 CPU
    uv run python scripts/train_l2.py --schema quad

Writes models/l2_<schema>.onnx and models/l2_<schema>_norm.json (normalization, residual median/MAD, and the
instant / EWMA z thresholds calibrated on the clean validation launches).
"""

from __future__ import annotations

import argparse
import os

import torch

from vbw.detector.data import clean_runs
from vbw.detector.l2 import MODELS_DIR
from vbw.detector.train import FitConfig, train_and_export_l2
from vbw.schema import SCHEMAS_BY_NAME


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", choices=["falcon9", "quad"], default="falcon9")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3, help="noise realizations per training launch (augmentation)")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    schema = SCHEMAS_BY_NAME[args.schema]
    train = clean_runs(schema, "train", tuple(range(args.seeds)))
    val = clean_runs(schema, "val", (100,))
    print(f"{schema.name}: {len(train)} train runs, {len(val)} val runs")
    cfg = FitConfig(epochs=args.epochs, hidden=args.hidden, stride=args.stride)
    l2, stats = train_and_export_l2(schema, train, val, cfg, MODELS_DIR)
    print("val z stats:", {k: round(v, 3) for k, v in stats.items()})
    print(f"thresholds: inst {l2.norm.inst_threshold:.2f}, ewma {l2.norm.ewma_threshold:.3f}; "
          f"onnx/torch max diff {l2.norm.meta['onnx_torch_max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()

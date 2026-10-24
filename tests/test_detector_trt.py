"""TensorRT backend: FP32 / FP16 engines of a small model agree with ONNX Runtime. Skipped without TensorRT + a GPU.

Run with the optional GPU packages:
    uv run --with tensorrt-cu12 --with "cuda-python>=12.6,<13" --with onnxconverter-common pytest tests/test_detector_trt.py
"""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
pytest.importorskip("cuda.bindings")
pytest.importorskip("onnxconverter_common")


def _gpu() -> bool:
    from cuda.bindings import runtime as rt

    err, n = rt.cudaGetDeviceCount()
    return int(err) == 0 and n > 0


pytestmark = pytest.mark.skipif(not _gpu(), reason="no CUDA GPU")


@pytest.fixture()
def tiny_onnx(tmp_path):
    import torch

    from vbw.detector.nets import export_onnx

    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4 * 3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1),
                              torch.nn.Flatten(0))
    path = tmp_path / "tiny.onnx"
    export_onnx(net.eval(), np.zeros((2, 4, 3), np.float32), path, "logit")
    return path


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_engine_matches_onnxruntime(tiny_onnx, tmp_path, precision):
    import onnxruntime as ort

    from vbw.detector.trt import TrtModel, build_engine, to_fp16

    src = to_fp16(tiny_onnx, tmp_path / "tiny_fp16.onnx") if precision == "fp16" else tiny_onnx
    layers = build_engine(src, tmp_path / f"tiny_{precision}.engine", "x", (4, 3), opt_batch=8, max_batch=64)
    assert layers.get(precision, 0) > 0                       # the engine really runs in the requested precision
    x = np.random.default_rng(0).normal(size=(32, 4, 3)).astype(np.float32)
    ref = ort.InferenceSession(str(tiny_onnx), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    model = TrtModel(tmp_path / f"tiny_{precision}.engine")
    atol = 2e-3 if precision == "fp32" else 3e-2   # "fp32" GEMMs run in TF32 on tensor cores (10-bit mantissa)
    try:
        np.testing.assert_allclose(model(x), ref, atol=atol)
        np.testing.assert_allclose(model(x[:1]), ref[:1], atol=atol)
    finally:
        model.close()

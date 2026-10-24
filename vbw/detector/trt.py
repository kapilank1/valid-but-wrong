"""TensorRT backend for the detector's ONNX models (L2 GRU, L3 discriminator), FP32 / FP16 / INT8.

TensorRT 11 networks are strongly typed: precision comes from the ONNX graph, not from builder flags. So:

* fp32: the exported model as is,
* fp16: the graph converted to half precision with float32 inputs/outputs kept (onnxconverter-common),
* int8: explicit Quantize/DequantizeLinear pairs from ONNX Runtime static quantization, calibrated on real flight
  windows (symmetric, per-channel weights). TensorRT fuses them into INT8 kernels; ops without Q/DQ (the GRU cell)
  stay in float.

Optional GPU dependencies (not in the default env, because they depend on the NVIDIA driver):

    uv run --with tensorrt-cu12 --with "cuda-python>=12.6,<13" --with onnxconverter-common python scripts/trt_bench.py

`cuda-python` must match the driver's CUDA major version (12.x for driver 576 / CUDA 12.9).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

PRECISIONS = ("fp32", "fp16", "int8")


def _check(result):
    """cuda-python calls return (err, *values); raise on error, return the values."""
    err, *values = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return values[0] if len(values) == 1 else values


# --------------------------------------------------------------------------- ONNX precision variants


def to_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
    onnx.save(model, dst)
    return dst


def to_int8(src: Path, dst: Path, calibration: np.ndarray, input_name: str = "x",
            op_types: tuple[str, ...] = ("Conv", "Gemm", "MatMul"), method: str = "minmax",
            keep_first_float: bool = True) -> Path:
    """Static INT8 (QDQ) quantization calibrated on `calibration` [N, ...] float32 inputs.

    method: "minmax", "percentile" or "entropy" (the histogram calibrators crash in ONNX Runtime on these graphs).
    keep_first_float: leave the first quantizable layer in float. Its input is the raw feature window (±10 range,
    mostly near 0), where a single INT8 scale is too coarse; this mixed precision is what TensorRT guides recommend.
    """
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, batch: int = 64):
            self._batches = iter([{input_name: data[i:i + batch]} for i in range(0, len(data), batch)])

        def get_next(self):
            return next(self._batches, None)

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    candidates = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type in op_types]
    exclude = candidates[:1] if keep_first_float else []
    quantize_static(str(pre), str(dst), Reader(calibration.astype(np.float32)), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=list(op_types), nodes_to_exclude=exclude,
                    calibrate_method={"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                                      "percentile": CalibrationMethod.Percentile}[method],
                    # TensorRT wants symmetric INT8 and float biases (it fuses them into the INT8 kernel itself).
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
    pre.unlink(missing_ok=True)
    return dst


# --------------------------------------------------------------------------- engine build


def build_engine(onnx_path: Path, engine_path: Path, input_name: str, sample_shape: tuple[int, ...],
                 opt_batch: int = 256, max_batch: int = 4096) -> dict:
    """Build a dynamic-batch engine and return layer precision counts from the engine inspector."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, *sample_shape), (opt_batch, *sample_shape), (max_batch, *sample_shape))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(plan))
    return layer_precisions(engine_path)


def layer_precisions(engine_path: Path) -> dict:
    """{precision: layer count} as TensorRT actually built it (from the engine inspector)."""
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    names = {"Float": "fp32", "Half": "fp16", "Int8": "int8", "FP8": "fp8", "BFloat16": "bf16"}
    counts: dict[str, int] = {}
    for layer in info.get("Layers", []):
        # Count layers by the type of data they produce; shape bookkeeping (Int32 / Int64 / Bool) is skipped.
        types = {t.get("Datatype") for t in layer.get("Outputs", []) if isinstance(t, dict)} if isinstance(layer, dict) else set()
        for dtype in types & names.keys():
            counts[names[dtype]] = counts.get(names[dtype], 0) + 1
    return counts


# --------------------------------------------------------------------------- runtime


class TrtModel:
    """A TensorRT engine with one float32 input and one float32 output, run on its own CUDA stream.

    Host buffers are pinned and device buffers are reused across calls (grown when a bigger batch arrives), so a
    call is: copy in, execute, copy out, synchronize. That is the honest per-frame cost for a streaming detector.
    """

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        from cuda.bindings import runtime as rt

        self._trt, self._rt = trt, rt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.stream = _check(rt.cudaStreamCreate())
        self._cap = 0
        self._bufs: dict[str, tuple[int, int, np.ndarray]] = {}   # name -> (device ptr, host ptr, host view)

    def _ensure(self, name: str, nbytes: int, shape: tuple[int, ...]) -> np.ndarray:
        rt = self._rt
        dev, host, view = self._bufs.get(name, (0, 0, np.zeros(0, np.float32)))
        if view.nbytes < nbytes:
            if dev:
                _check(rt.cudaFree(dev))
                _check(rt.cudaFreeHost(host))
            dev = _check(rt.cudaMalloc(nbytes))
            host = _check(rt.cudaMallocHost(nbytes))
            view = np.ctypeslib.as_array((ctypes.c_float * (nbytes // 4)).from_address(int(host)))
            self._bufs[name] = (dev, host, view)
        return view[: int(np.prod(shape))].reshape(shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rt = self._rt
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output))
        host_in = self._ensure(self.input, x.nbytes, x.shape)
        host_out = self._ensure(self.output, int(np.prod(out_shape)) * 4, out_shape)
        host_in[...] = x
        d_in, d_out = self._bufs[self.input][0], self._bufs[self.output][0]
        self.context.set_tensor_address(self.input, int(d_in))
        self.context.set_tensor_address(self.output, int(d_out))
        _check(rt.cudaMemcpyAsync(d_in, host_in.ctypes.data, x.nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                  self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execution failed")
        _check(rt.cudaMemcpyAsync(host_out.ctypes.data, d_out, host_out.nbytes,
                                  rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        _check(rt.cudaStreamSynchronize(self.stream))
        return host_out.copy()

    def close(self) -> None:
        for dev, host, _ in self._bufs.values():
            self._rt.cudaFree(dev)
            self._rt.cudaFreeHost(host)
        self._bufs.clear()
        self._rt.cudaStreamDestroy(self.stream)

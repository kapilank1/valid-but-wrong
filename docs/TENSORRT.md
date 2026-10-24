# TensorRT: scoring a fleet on one laptop GPU

The detector's learned layers (L2 GRU next-step predictor, L3 forgery discriminator) are exported to ONNX and run on
ONNX Runtime (CPU) in the live pipeline. `scripts/trt_bench.py` builds TensorRT engines for both models in FP32, FP16
and INT8 and measures what a GPU buys. All numbers below are from `results/trt_bench.json`.

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0, 8 GB, driver 576.76), TensorRT 11.4, CUDA 12
runtime; CPU baseline: AMD Ryzen AI 7 350 (8 cores) with ONNX Runtime.

```powershell
uv run --with tensorrt-cu12 --with "cuda-python>=12.6,<13" --with onnxconverter-common python scripts/trt_bench.py
```

## How the precisions are made

TensorRT 11 networks are *strongly typed*: there are no FP16 / INT8 builder flags and no implicit INT8 calibrator.
Precision lives in the ONNX graph (`vbw/detector/trt.py`):

| Precision | How |
|---|---|
| FP32 | the exported ONNX as is (TensorRT runs FP32 GEMMs as TF32 on tensor cores) |
| FP16 | graph converted to half precision, float32 inputs/outputs kept (`onnxconverter-common`) |
| INT8 | explicit Quantize/DequantizeLinear pairs from ONNX Runtime static quantization: symmetric, per-channel weights, float biases, calibrated on 4,000 (L2) / 3,000 (L3) real training windows, first layer kept in float |

The engine inspector reports what actually runs: L2 FP16 = 31 FP16 layers + 1 FP32; L2 INT8 = only 2 INT8 layers
(TensorRT keeps the GRU cell in float); L3 INT8 = 6 INT8 + 5 FP32.

## Results

Fleet throughput, one batch = one frame from every vehicle stream (p50, copies in and out included):

| Streams per batch | Model | ONNX Runtime CPU | TRT FP32 | **TRT FP16** | TRT INT8 |
|---|---|---|---|---|---|
| 4,096 | L2 GRU | 46.1 ms | 3.0 ms | **1.6 ms** | 2.7 ms |
| 16,384 | L2 GRU | 203.9 ms | 13.4 ms | **6.1 ms** | 12.0 ms |
| 4,096 | L3 CNN | 25.6 ms | 2.8 ms | **2.3 ms** | 2.5 ms |
| 16,384 | L3 CNN | 70.3 ms | 11.7 ms | **9.1 ms** | 10.5 ms |

At FP16, L2 + L3 for **16,384 streams take ~15 ms, inside one 20 ms frame at 50 Hz**; the CPU sustains 1,024 streams
(L2) at 50 Hz. That is the learned layers only: the L1 physics rules and the Python pipeline are not included.

Accuracy against the FP32 ONNX Runtime reference, on held-out flights (30,844 L2 frames; 9,008 L3 windows of clean
held-out flight plus attack windows):

| | L2: max / p99 z-score error | L2 spike decisions changed | L3: max / p99 logit error | L3 forged/clean decisions changed |
|---|---|---|---|---|
| TRT FP32 | 0.02 / 0.01 σ | 0 | 0.05 / 0.02 | 0 |
| **TRT FP16** | 0.10 / 0.03 σ | **0** | 0.21 / 0.06 | **0** |
| TRT INT8 | 1.73 / 1.33 σ | 0 | 6.0 / 1.84 | 1.2% |

## What we learned

- **FP16 is the right precision here**: 33× (L2) and 7.7× (L3) faster than the CPU at 16,384 streams with no change
  to any detection decision.
- **INT8 doesn't pay off on models this small.** The GRU cell can't be quantized, the dense layers are tiny, and the
  Q/DQ overhead makes INT8 slower than FP16 while costing accuracy (1.2% of L3 decisions flip). Min/max calibration
  was used; ONNX Runtime's histogram calibrators (percentile, entropy) crash on these graphs.
- **One stream at a time, the GPU loses**: a single window takes ~0.2–0.6 ms on TensorRT (launch and copy overhead
  under Windows WDDM) vs ~0.04 ms on ONNX Runtime CPU. The live single-vehicle demo therefore stays on CPU; the GPU
  is for scoring many vehicles at once.

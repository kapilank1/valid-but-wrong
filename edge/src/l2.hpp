// L2 learned next-step predictor: the GRU (ONNX Runtime) and its robust residual z-scores.
// Mirrors vbw/detector/l2.py (l2_inputs, l2_target, L2Model.zscore_last, residuals_to_z, ewma_update, l2_ratio).
//
// Frame k is judged by predicting its normalized delta (x_k - x_{k-1}) / dstd from the previous `window` frames;
// the prediction error per channel becomes a robust z-score with the clean residual median / MAD.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vbw {

struct Config;

constexpr double Z_CLIP = 50.0;  // default when the config has no l2.z_clip

struct L2Norm {
  int window = 20;
  std::vector<double> mean, stdev, dstd, res_median, res_mad;  // per channel (stdev = Python's `std`)
  double inst_threshold = 8.0;   // max|z| that counts as ratio 1
  double ewma_alpha = 0.1;
  double ewma_threshold = 3.0;   // max|EWMA z| that counts as ratio 1
  double z_clip = Z_CLIP;
  double mad_floor = 1e-6;       // np.maximum(res_mad, 1e-6)
  static L2Norm from(const Config& cfg);  // l2.window, l2.mean, ... (throws if a per-channel array is missing)
};

// [W+1][C] raw frames (row-major) -> [W][2C] model input: normalized values of the last W frames, then their
// deltas / dstd. Computed in double and cast to float32 at the end, like numpy's .astype(np.float32).
void l2_inputs(const double* X, const L2Norm& norm, std::size_t channels, float* out);

// Normalized delta the model should predict: float32((x_next - x_prev) / dstd).
void l2_target(const double* x_prev, const double* x_next, const L2Norm& norm, std::size_t channels, float* out);

// ewma = (1 - alpha) * ewma + alpha * z
void ewma_update(std::vector<double>& state, const std::vector<double>& z, double alpha);

// max(max|z| / inst_threshold, max|ewma| / ewma_threshold)
double l2_ratio(const std::vector<double>& z, const std::vector<double>& ewma, const L2Norm& norm);

// ONNX Runtime inference plus z-scoring. Stateless apart from reusable I/O buffers (so not thread-safe per
// instance; give each detector its own L2Model). Sessions are shared process-wide per model path.
class L2Model {
 public:
  explicit L2Model(const Config& cfg);                 // model path = cfg.path("l2.onnx")
  L2Model(const std::string& onnx_path, L2Norm norm);
  ~L2Model();
  L2Model(const L2Model&) = delete;
  L2Model& operator=(const L2Model&) = delete;

  int needs() const { return norm_.window + 2; }       // frames of history needed to score the newest frame
  int window() const { return norm_.window; }
  std::size_t channels() const { return norm_.mean.size(); }
  const L2Norm& norm() const { return norm_; }

  // z-scores of the newest frame of X [needs][C] row-major (if X holds more rows, its last `needs` are used).
  std::vector<double> zscore_last(const std::vector<double>& X) const;
  // Allocation-free variant: X points at exactly needs() rows of C values; writes C z-scores.
  void zscore_last(const double* X, double* z) const;

  // The raw model output (float32 normalized delta prediction) of the last zscore_last call.
  const std::vector<float>& last_prediction() const;

 private:
  struct Impl;
  L2Norm norm_;
  std::vector<double> mad_;  // max(res_mad, 1e-6)
  std::unique_ptr<Impl> impl_;
};

// True when cfg names an L2 model file that exists (BaseDetector runs L1-only otherwise, like Python).
bool l2_available(const Config& cfg);

}  // namespace vbw

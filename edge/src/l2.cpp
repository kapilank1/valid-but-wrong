#include "l2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "config.hpp"

namespace vbw {

namespace {

using OrtPath = std::basic_string<ORTCHAR_T>;

// ORT on Windows takes wide-char paths; config paths are UTF-8.
OrtPath to_ort_path(const std::string& utf8) {
#ifdef _WIN32
  if (utf8.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
  if (n <= 0) throw std::runtime_error("l2: invalid UTF-8 model path: " + utf8);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), w.data(), n);
  return w;
#else
  return utf8;
#endif
}

bool file_exists(const std::string& utf8) {
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path(to_ort_path(utf8)), ec);
}

// One process-wide environment. Deliberately leaked so it outlives every session regardless of static
// destruction order (sessions may be held by detectors with static storage duration).
Ort::Env& ort_env() {
  static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "vbw_edge");
  return *env;
}

// Sessions are immutable after creation and Session::Run is thread-safe, so every detector of a process shares
// one session per model file (a new vehicle doesn't pay the model load on its first frame).
std::shared_ptr<Ort::Session> shared_session(const std::string& path) {
  static std::mutex* mu = new std::mutex;
  static auto* cache = new std::map<std::string, std::shared_ptr<Ort::Session>>;
  std::lock_guard<std::mutex> lock(*mu);
  auto it = cache->find(path);
  if (it != cache->end()) return it->second;
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);   // streaming batch of 1: threads only add overhead (as in Python)
  opts.SetInterOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  // Python's default
  const OrtPath wpath = to_ort_path(path);
  auto session = std::make_shared<Ort::Session>(ort_env(), wpath.c_str(), opts);
  (*cache)[path] = session;
  return session;
}

void require_size(const std::vector<double>& v, size_t n, const char* name) {
  if (v.size() != n)
    throw std::runtime_error(std::string("l2: ") + name + " has " + std::to_string(v.size()) + " values, expected " +
                             std::to_string(n));
}

}  // namespace

// --------------------------------------------------------------------------- norm and pure helpers

L2Norm L2Norm::from(const Config& cfg) {
  L2Norm n;
  n.window = static_cast<int>(cfg.num_or("l2.window", n.window));
  n.mean = cfg.vec("l2.mean");
  n.stdev = cfg.vec("l2.std");
  n.dstd = cfg.vec("l2.dstd");
  n.res_median = cfg.vec("l2.res_median");
  n.res_mad = cfg.vec("l2.res_mad");
  n.inst_threshold = cfg.num_or("l2.inst_threshold", n.inst_threshold);
  n.ewma_alpha = cfg.num_or("l2.ewma_alpha", n.ewma_alpha);
  n.ewma_threshold = cfg.num_or("l2.ewma_threshold", n.ewma_threshold);
  n.z_clip = cfg.num_or("l2.z_clip", n.z_clip);
  n.mad_floor = cfg.num_or("l2.mad_floor", n.mad_floor);
  return n;
}

void l2_inputs(const double* X, const L2Norm& norm, size_t channels, float* out) {
  const size_t C = channels, W = static_cast<size_t>(norm.window);
  for (size_t k = 1; k <= W; ++k) {
    const double* cur = X + k * C;
    const double* prev = cur - C;
    float* row = out + (k - 1) * 2 * C;
    for (size_t c = 0; c < C; ++c) {
      row[c] = static_cast<float>((cur[c] - norm.mean[c]) / norm.stdev[c]);
      row[C + c] = static_cast<float>((cur[c] - prev[c]) / norm.dstd[c]);
    }
  }
}

void l2_target(const double* x_prev, const double* x_next, const L2Norm& norm, size_t channels, float* out) {
  for (size_t c = 0; c < channels; ++c) out[c] = static_cast<float>((x_next[c] - x_prev[c]) / norm.dstd[c]);
}

void ewma_update(std::vector<double>& state, const std::vector<double>& z, double alpha) {
  for (size_t c = 0; c < state.size(); ++c) state[c] = (1.0 - alpha) * state[c] + alpha * z[c];
}

namespace {
// float(np.max(np.abs(v))): NaN propagates.
double max_abs(const std::vector<double>& v) {
  double m = -INFINITY;
  for (double x : v) {
    const double a = std::fabs(x);
    if (std::isnan(a)) return a;
    if (a > m) m = a;
  }
  return m;
}
}  // namespace

double l2_ratio(const std::vector<double>& z, const std::vector<double>& ewma, const L2Norm& norm) {
  const double inst = max_abs(z) / norm.inst_threshold;
  const double slow = max_abs(ewma) / norm.ewma_threshold;
  return slow > inst ? slow : inst;  // Python max(inst, slow)
}

bool l2_available(const Config& cfg) {
  if (!cfg.has("l2.onnx")) return false;
  try {
    return file_exists(cfg.path("l2.onnx"));
  } catch (const std::exception&) {
    return false;
  }
}

// --------------------------------------------------------------------------- model

struct L2Model::Impl {
  std::shared_ptr<Ort::Session> session;
  Ort::MemoryInfo mem{nullptr};
  Ort::RunOptions run_opts;
  std::vector<float> in, out, target;
  std::array<int64_t, 3> in_shape{};
  std::array<int64_t, 2> out_shape{};
  Ort::Value in_val{nullptr}, out_val{nullptr};
  const char* in_name = "x";
  const char* out_name = "delta";
};

L2Model::L2Model(const Config& cfg) : L2Model(cfg.path("l2.onnx"), L2Norm::from(cfg)) {}

L2Model::L2Model(const std::string& onnx_path, L2Norm norm) : norm_(std::move(norm)), impl_(std::make_unique<Impl>()) {
  const size_t C = norm_.mean.size();
  if (C == 0) throw std::runtime_error("l2: no channels in the norm");
  if (norm_.window < 1) throw std::runtime_error("l2: window must be >= 1");
  require_size(norm_.stdev, C, "l2.std");
  require_size(norm_.dstd, C, "l2.dstd");
  require_size(norm_.res_median, C, "l2.res_median");
  require_size(norm_.res_mad, C, "l2.res_mad");
  mad_.resize(C);
  for (size_t c = 0; c < C; ++c) mad_[c] = std::max(norm_.res_mad[c], norm_.mad_floor);  // np.maximum(res_mad, 1e-6)

  if (!file_exists(onnx_path)) throw std::runtime_error("l2: model file not found: " + onnx_path);
  Impl& m = *impl_;
  m.session = shared_session(onnx_path);

  // Check the I/O contract: input "x" [batch, window, 2C], output "delta" [batch, C].
  Ort::AllocatorWithDefaultOptions alloc;
  if (m.session->GetInputCount() != 1 || m.session->GetOutputCount() != 1)
    throw std::runtime_error("l2: expected one input and one output in " + onnx_path);
  const std::string in_name = m.session->GetInputNameAllocated(0, alloc).get();
  const std::string out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  if (in_name != "x" || out_name != "delta")
    throw std::runtime_error("l2: expected input 'x' and output 'delta', got '" + in_name + "' / '" + out_name + "'");
  const auto in_dims = m.session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto out_dims = m.session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const int64_t W = norm_.window, C2 = static_cast<int64_t>(2 * C), CC = static_cast<int64_t>(C);
  if (in_dims.size() != 3 || (in_dims[1] > 0 && in_dims[1] != W) || (in_dims[2] > 0 && in_dims[2] != C2))
    throw std::runtime_error("l2: model input shape does not match [batch, window, 2C]");
  if (out_dims.size() != 2 || (out_dims[1] > 0 && out_dims[1] != CC))
    throw std::runtime_error("l2: model output shape does not match [batch, C]");

  // Reused I/O buffers: tensors wrap them once, every call writes in place.
  m.mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  m.in.assign(static_cast<size_t>(W * C2), 0.0f);
  m.out.assign(C, 0.0f);
  m.target.assign(C, 0.0f);
  m.in_shape = {1, W, C2};
  m.out_shape = {1, CC};
  m.in_val = Ort::Value::CreateTensor<float>(m.mem, m.in.data(), m.in.size(), m.in_shape.data(), m.in_shape.size());
  m.out_val = Ort::Value::CreateTensor<float>(m.mem, m.out.data(), m.out.size(), m.out_shape.data(), m.out_shape.size());
}

L2Model::~L2Model() = default;

const std::vector<float>& L2Model::last_prediction() const { return impl_->out; }

void L2Model::zscore_last(const double* X, double* z) const {
  Impl& m = *impl_;
  const size_t C = channels(), W = static_cast<size_t>(norm_.window);
  // span = X[-needs:]; pred = predict(l2_inputs(span[None, :-1]))[0]
  l2_inputs(X, norm_, C, m.in.data());
  m.session->Run(m.run_opts, &m.in_name, &m.in_val, 1, &m.out_name, &m.out_val, 1);
  // residuals_to_z(l2_target(span[-2], span[-1]) - pred): float32 subtraction, then float64 z
  l2_target(X + W * C, X + (W + 1) * C, norm_, C, m.target.data());
  const double zc = norm_.z_clip;
  for (size_t c = 0; c < C; ++c) {
    const float residual = m.target[c] - m.out[c];
    double v = (static_cast<double>(residual) - norm_.res_median[c]) / mad_[c];
    if (v < -zc) v = -zc;  // np.clip (NaN passes through)
    else if (v > zc) v = zc;
    z[c] = v;
  }
}

std::vector<double> L2Model::zscore_last(const std::vector<double>& X) const {
  const size_t C = channels(), need = static_cast<size_t>(needs());
  if (X.size() % C != 0 || X.size() / C < need)
    throw std::runtime_error("l2: zscore_last needs at least " + std::to_string(need) + " frames");
  std::vector<double> z(C);
  zscore_last(X.data() + (X.size() - need * C), z.data());
  return z;
}

}  // namespace vbw

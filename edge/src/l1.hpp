// L1 physics-consistency rules: exact port of vbw/detector/physics.py (FrameBuffer, check_rule, L1Checker and the
// Falcon 9 rules). Every rule computes only the residual of the NEWEST frame, over exactly the window the Python
// uses (buffer.last(lookback + 1)) and with the same floating-point operation order, so ratios match numpy to the
// last bit or so (golden parity: edge/tests/test_l1.cpp).
//
// Rules are configured from the edge config (edge/SPEC.md section 2):
//   l1.rules                  rule names, in order (each must be implemented for the config's `schema`)
//   l1.threshold.<rule>       calibrated threshold      l1.lookback.<rule>   frames of history before the newest
//   l1.params.<dotted.path>   flattened L1 params dict  l1.const.<NAME>      module constants (F9_WINDOW, G0, ...)
// Missing params/consts fall back to the Python defaults (`params.get(key, default)` / module constants).
//
// Numerics: build with IEEE semantics (MSVC /fp:precise, the default; never /fp:fast; GCC/Clang need
// -ffp-contract=off so a*b+c is not fused). Transcendentals come from the C runtime (numpy on Windows uses the same
// UCRT functions, verified bit-identical for sin/atan2/exp/hypot).
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "config.hpp"

namespace vbw {

// Fixed-capacity buffer of frames whose last n frames are always contiguous (like physics.FrameBuffer: 2x storage,
// the newest half is shifted down when the end is reached; amortized O(1) append).
class FrameBuffer {
 public:
  FrameBuffer(size_t channels, size_t capacity);
  void clear();
  void append(const std::vector<double>& x, double t);
  void append(const double* x, double t);
  size_t size() const;
  size_t channels() const { return channels_; }
  size_t capacity() const { return capacity_; }
  // Copies the last n (clamped to size()) frames: X row-major [n][channels], t [n].
  void last(size_t n, std::vector<double>& X, std::vector<double>& t) const;
  // Zero-copy views of the last n (clamped to size()) frames; valid until the next append()/clear().
  const double* last_X(size_t n) const;
  const double* last_t(size_t n) const;

 private:
  size_t channels_, capacity_, end_ = 0;
  std::vector<double> X_, t_;
};

struct L1Result {
  std::vector<double> ratios;   // signed residual / threshold per rule, in rule order (0 while not evaluable)
  double max_ratio = 0;         // max |ratio|
  std::vector<int> violated;    // indices of violated rules, worst first (Python's sorted L1Result.violations)
  std::string message;          // message of the worst violation ("" when none)
};

// One window handed to a rule: the last n frames, X row-major [n][C], t [n].
struct L1Window {
  const double* X;
  const double* t;
  size_t n;
  size_t C;
  double at(size_t k, size_t c) const { return X[k * C + c]; }
};

// A rule implementation. residual() returns the LAST element of the Python residual array over the window (NaN
// where Python would produce NaN); message() ports the Python message function.
struct L1Rule {
  std::string name;
  size_t lookback = 0;
  double threshold = 0;
  std::function<double(const L1Window&)> residual;
  std::function<std::string(double value, double threshold, const L1Window&)> message;
};

class L1Checker {
 public:
  explicit L1Checker(const Config& cfg);
  void reset();
  L1Result update(const std::vector<double>& x, double t);   // append + evaluate every rule on the newest frame
  void update(const std::vector<double>& x, double t, L1Result& out);   // same, reusing out's storage
  const FrameBuffer& buffer() const { return buffer_; }
  const std::vector<std::string>& rule_names() const { return names_; }
  const std::vector<double>& thresholds() const { return thresholds_; }
  const std::vector<size_t>& lookbacks() const { return lookbacks_; }

  // For tests: residual value of rule i on the current buffer (NaN when not evaluable) and its message for a
  // given value (Python rule.message(value, threshold, X, params) on the current window).
  double residual(size_t i) const;
  std::string message(size_t i, double value) const;

 private:
  bool window_for(size_t i, L1Window& w) const;

  std::vector<L1Rule> rules_;
  std::vector<std::string> names_;
  std::vector<double> thresholds_;
  std::vector<size_t> lookbacks_;
  FrameBuffer buffer_;
  struct Cand { int rule; double key; double value; };
  std::vector<Cand> cands_;   // scratch
};

// Builds the configured rules (schema from cfg `schema`). Throws std::runtime_error for an unknown schema or rule.
std::vector<L1Rule> make_l1_rules(const Config& cfg);

// numpy / Python semantics helpers, exposed for unit tests.
namespace l1 {
double np_median(double* v, size_t m);   // np.median of v[0..m) (reorders v); NaN if any NaN; even m: (lo + hi) / 2
double np_maximum(double a, double b);   // np.maximum (NaN-propagating)
double np_minimum(double a, double b);
double np_remainder(double a, double b); // Python/numpy float %: result has the sign of b
double wrap_deg(double x);               // (x + 180) % 360 - 180
double degrees(double rad);              // np.degrees: rad * (180 / pi)
// np.interp(x, xp, fp, left, right) for strictly increasing xp; left/right default to fp[0] / fp[-1].
double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp);
double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp, double left, double right);
// cumtrapz(y, t) (cumulative trapezoid starting at 0) for a strided y and contiguous t; writes n values to out.
void cumtrapz(const double* y, size_t y_stride, const double* t, size_t n, double* out);
// Python format(v, ".{prec}f") (correctly rounded, ties to even, "-0" for negative values rounding to zero).
std::string fmt_fixed(double v, int prec);
// Python round(v, ndigits).
double py_round(double v, int ndigits);
}  // namespace l1

}  // namespace vbw

// Exact port of the L1 physics rules in vbw/detector/physics.py. See l1.hpp.
//
// Each Falcon 9 rule below mirrors one f9_* function. Comments quote the Python expression being reproduced; the
// operation order is kept so results match numpy bit-for-bit (or within an ulp or two of a C runtime function).
#include "l1.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>

// No a*b+c contraction into FMA (numpy's loops are not contracted); no auto-vectorized libm calls (MSVC's vector
// math routines can differ from the scalar CRT functions numpy uses by an ulp).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#define VBW_SCALAR_LOOP __pragma(loop(no_vector))
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#define VBW_SCALAR_LOOP
#else
#define VBW_SCALAR_LOOP
#endif

namespace vbw {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.141592653589793;   // math.pi / NPY_PI

}  // namespace

// ============================================================================================ FrameBuffer

FrameBuffer::FrameBuffer(size_t channels, size_t capacity)
    : channels_(channels), capacity_(capacity), X_(2 * capacity * channels, 0.0), t_(2 * capacity, 0.0) {
  if (capacity == 0) throw std::runtime_error("FrameBuffer: capacity must be > 0");
}

void FrameBuffer::clear() { end_ = 0; }

size_t FrameBuffer::size() const { return std::min(end_, capacity_); }

void FrameBuffer::append(const std::vector<double>& x, double t) {
  if (x.size() != channels_) {
    throw std::runtime_error("FrameBuffer::append: got " + std::to_string(x.size()) + " values, expected " +
                             std::to_string(channels_));
  }
  append(x.data(), t);
}

void FrameBuffer::append(const double* x, double t) {
  if (end_ == 2 * capacity_) {   // shift the newest half down; amortized O(1)
    std::copy(X_.begin() + static_cast<std::ptrdiff_t>(capacity_ * channels_), X_.end(), X_.begin());
    std::copy(t_.begin() + static_cast<std::ptrdiff_t>(capacity_), t_.end(), t_.begin());
    end_ = capacity_;
  }
  std::copy(x, x + channels_, X_.begin() + static_cast<std::ptrdiff_t>(end_ * channels_));
  t_[end_] = t;
  ++end_;
}

const double* FrameBuffer::last_X(size_t n) const {
  n = std::min(n, size());
  return X_.data() + (end_ - n) * channels_;
}

const double* FrameBuffer::last_t(size_t n) const {
  n = std::min(n, size());
  return t_.data() + (end_ - n);
}

void FrameBuffer::last(size_t n, std::vector<double>& X, std::vector<double>& t) const {
  n = std::min(n, size());
  const double* px = last_X(n);
  const double* pt = last_t(n);
  X.assign(px, px + n * channels_);
  t.assign(pt, pt + n);
}

// ============================================================================================ helpers

namespace l1 {

double np_median(double* v, size_t m) {
  if (m == 0) return kNaN;
  for (size_t i = 0; i < m; ++i) {
    if (std::isnan(v[i])) return kNaN;   // np.median returns NaN if any element is NaN
  }
  const size_t h = m / 2;
  std::nth_element(v, v + h, v + m);
  const double hi = v[h];
  if (m % 2 == 1) return hi;              // mean of the single middle element
  const double lo = *std::max_element(v, v + h);
  return (lo + hi) / 2.0;                 // np.mean of the two middle elements: (a + b) / 2
}

double np_maximum(double a, double b) { return (a >= b || std::isnan(a)) ? a : b; }

double np_minimum(double a, double b) { return (a <= b || std::isnan(a)) ? a : b; }

double np_remainder(double a, double b) {
  // npy_remainder -> npy_divmod: fmod adjusted to the sign of the divisor (Python convention).
  double mod = std::fmod(a, b);
  if (b == 0) return mod;
  if (mod != 0) {
    if ((b < 0) != (mod < 0)) mod += b;
  } else {
    mod = std::copysign(0.0, b);
  }
  return mod;
}

double wrap_deg(double x) { return np_remainder(x + 180.0, 360.0) - 180.0; }

double degrees(double rad) {
  constexpr double kRad2Deg = 180.0 / kPi;   // npy_rad2deg: x * (180.0 / NPY_PI)
  return rad * kRad2Deg;
}

double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp, double left,
                 double right) {
  const size_t n = xp.size();
  if (n == 0 || fp.size() != n) throw std::runtime_error("np_interp: xp and fp must be non-empty and equal length");
  if (n == 1) return (x < xp[0]) ? left : ((x > xp[0]) ? right : fp[0]);
  if (std::isnan(x)) return x;
  // binary_search_with_guess (numpy compiled_base.c); for strictly increasing xp every path returns the largest j
  // with xp[j] <= x, -1 below the grid, n above it.
  if (x > xp[n - 1]) return right;
  if (x < xp[0]) return left;
  const size_t j = static_cast<size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
  if (j == n - 1) return fp[j];
  if (xp[j] == x) return fp[j];   // avoid potential non-finite interpolation
  const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  double r = slope * (x - xp[j]) + fp[j];
  if (std::isnan(r)) {   // if we get nan in one direction, try the other
    r = slope * (x - xp[j + 1]) + fp[j + 1];
    if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
  }
  return r;
}

double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
  if (fp.empty()) throw std::runtime_error("np_interp: empty fp");
  return np_interp(x, xp, fp, fp.front(), fp.back());
}

void cumtrapz(const double* y, size_t y_stride, const double* t, size_t n, double* out) {
  // steps = 0.5 * (y[1:] + y[:-1]) * np.diff(t); out = [0, cumsum(steps)] (np.cumsum is a sequential running sum)
  if (n == 0) return;
  out[0] = 0.0;
  if (n == 1) return;
  double acc = (0.5 * (y[y_stride] + y[0])) * (t[1] - t[0]);
  out[1] = acc;
  for (size_t k = 2; k < n; ++k) {
    acc = acc + (0.5 * (y[k * y_stride] + y[(k - 1) * y_stride])) * (t[k] - t[k - 1]);
    out[k] = acc;
  }
}

std::string fmt_fixed(double v, int prec) {
  char buf[400];
  const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, prec);
  if (r.ec != std::errc()) return "?";
  return std::string(buf, r.ptr);
}

double py_round(double v, int ndigits) {
  // CPython float.__round__: correctly rounded decimal string (ties to even), parsed back.
  if (!std::isfinite(v)) return v;
  char buf[400];
  const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, ndigits);
  if (r.ec != std::errc()) return v;
  double out = v;
  std::from_chars(buf, r.ptr, out);
  return out;
}

}  // namespace l1

// ============================================================================================ Falcon 9 rules

namespace {

using l1::np_maximum;
using l1::np_median;
using l1::np_minimum;

// Literals that appear inline in physics.py (not module constants, so they are not in the config).
constexpr double kGravityMinAlt = -1000.0;       // gravity(): np.maximum(alt, -1000.0)
constexpr double kExpRho0 = 1.225;               // air_density() fallback: 1.225 * exp(-max(alt, 0) / 8500)
constexpr double kExpScaleHeight = 8500.0;
constexpr long kJerkSmooth = 5;                  // f9_jerk_envelope: rolling_median(accel, 5)
constexpr size_t kJerkLag = 10;                  //                   lagdiff(.., 10)
constexpr double kFrozenMinSpeed = 5.0;          // f9_frozen_channel: speed > 5.0
constexpr double kFrozenMinQ = 50.0;             //                    predicted_q > 50.0

long as_int(double v, const std::string& key) {
  if (!std::isfinite(v) || v != std::floor(v)) throw std::runtime_error("L1 config: " + key + " must be an integer");
  return static_cast<long>(v);
}

bool has_prefix(const Config& cfg, const std::string& prefix) { return !cfg.keys_with_prefix(prefix).empty(); }

// Shared state of the Falcon 9 rules: channel indices, params, constants and scratch buffers.
struct F9 {
  std::vector<std::string> channels;
  size_t C = 0, speed = 0, altitude = 0, vz = 0, vx = 0, downrange = 0, accel = 0, angle = 0, q = 0;

  // module constants (l1.const.*)
  double rate_hz = 10.0, min_speed = 50.0, min_flight_speed = 10.0, min_q_altitude = 200.0;
  double g0 = 9.80665, r_earth = 6371000.0;
  long long_window = 600;
  // params.get(...) with module defaults
  long window = 100, smooth = 10, q_smooth = 5;
  double q_floor = 1000.0, frozen_cap = 50.0;
  bool has_atm = false;
  std::vector<double> atm_alt, atm_log_rho;
  bool has_cor = false;
  std::vector<double> cor_t, alt_lo, alt_hi, spd_lo, spd_hi;
  double alt_floor = 0, spd_floor = 0, t_lo_end = 0;

  // scratch (grown on demand, never shrunk)
  std::vector<double> cum, med, kin, run;

  double* scratch(std::vector<double>& v, size_t n) {
    if (v.size() < n) v.resize(n);
    return v.data();
  }

  // gravity(alt) = G0 * (R_EARTH / (R_EARTH + np.maximum(alt, -1000.0))) ** 2   (array ** 2 is np.square)
  double gravity(double alt) const {
    const double r = r_earth / (r_earth + np_maximum(alt, kGravityMinAlt));
    return g0 * (r * r);
  }

  double air_density(double alt) const {
    if (has_atm) {
      // rho = np.exp(np.interp(alt, grid, log_rho)); np.where(alt > grid[-1], 0.0, rho)
      const double rho = std::exp(l1::np_interp(alt, atm_alt, atm_log_rho));
      return (alt > atm_alt.back()) ? 0.0 : rho;
    }
    return kExpRho0 * std::exp(-np_maximum(alt, 0.0) / kExpScaleHeight);
  }

  // predicted_q = 0.5 * air_density(alt) * speed ** 2
  double predicted_q(const L1Window& w, size_t k) const {
    const double v = w.at(k, speed);
    return (0.5 * air_density(w.at(k, altitude))) * (v * v);
  }

  // Last element of rolling_median(f, m) over the window (f evaluated at frame index k).
  template <class Fn>
  double rolling_median_last(size_t N, long m, Fn&& f) {
    if (m <= 1) return f(N - 1);
    if (N < static_cast<size_t>(m)) return kNaN;
    double* v = scratch(med, static_cast<size_t>(m));
    for (long i = 0; i < m; ++i) v[i] = f(N - static_cast<size_t>(m) + static_cast<size_t>(i));
    return np_median(v, static_cast<size_t>(m));
  }

  // Last element of integrated_mismatch(level, rate, t, w, m):
  //   offset = rolling_median(level - cumtrapz(rate, t), m); lagdiff(offset, w) / lagdiff(t, w)
  double integrated_mismatch_last(const L1Window& W, size_t level_ch, const double* rate, size_t rate_stride,
                                  long w, long m) {
    const size_t N = W.n;
    if (w <= 0 || N <= static_cast<size_t>(w)) return kNaN;   // lagdiff: all NaN when the window is too short
    double* c = scratch(cum, N);
    l1::cumtrapz(rate, rate_stride, W.t, N, c);
    auto offset = [&](size_t j) -> double {
      if (m <= 1) return W.at(j, level_ch) - c[j];
      const size_t mm = static_cast<size_t>(m);
      if (N < mm || j + 1 < mm) return kNaN;   // NaN padding of rolling_median
      double* v = scratch(med, mm);
      for (size_t i = 0; i < mm; ++i) {
        const size_t k = j + 1 - mm + i;
        v[i] = W.at(k, level_ch) - c[k];
      }
      return np_median(v, mm);
    };
    const size_t wl = static_cast<size_t>(w);
    const double a = offset(N - 1);
    const double b = offset(N - 1 - wl);
    const double elapsed = W.t[N - 1] - W.t[N - 1 - wl];
    return (a - b) / elapsed;
  }

  // frozen_run_length(X[-n_last:], cap)[-1] for every channel -> run[0..C)
  const double* frozen_run_last(const L1Window& W, size_t n_last, double cap) {
    double* r = scratch(run, C);
    std::fill(r, r + C, 0.0);
    const size_t n = std::min(n_last, W.n);
    const size_t k0 = W.n - n;
    for (size_t k = 0; k < n; ++k) {
      for (size_t ch = 0; ch < C; ++ch) {
        const bool same = k > 0 && (W.at(k0 + k, ch) - W.at(k0 + k - 1, ch) == 0);
        r[ch] = same ? np_minimum(r[ch] + 1.0, cap) : 0.0;
      }
    }
    return r;
  }

  // ---------------------------------------------------------------------------------------- residuals

  double vz_vs_altitude(const L1Window& W) {
    return integrated_mismatch_last(W, altitude, W.X + vz, C, window, smooth);
  }

  double vz_vs_altitude_long(const L1Window& W) {
    return integrated_mismatch_last(W, altitude, W.X + vz, C, long_window, smooth);
  }

  double vx_vs_downrange(const L1Window& W) {
    return integrated_mismatch_last(W, downrange, W.X + vx, C, window, smooth);
  }

  double accel_vs_speed(const L1Window& W) {
    const size_t N = W.n;
    // in_flight = rolling_min(speed, w + m) > F9_MIN_FLIGHT_SPEED; np.where(in_flight, mismatch, 0.0).
    // Evaluated first: when not in flight the result is 0.0 whatever the mismatch is.
    const long wm = window + smooth;
    if (wm <= 0 || N < static_cast<size_t>(wm)) return 0.0;   // NaN > x is False
    double mn = W.at(N - static_cast<size_t>(wm), speed);
    for (size_t k = N - static_cast<size_t>(wm) + 1; k < N; ++k) mn = np_minimum(mn, W.at(k, speed));
    if (!(mn > min_flight_speed)) return 0.0;
    // kinematic = accel - gravity(alt) * sin(arctan2(vz, |vx|))
    double* kn = scratch(kin, N);
    VBW_SCALAR_LOOP
    for (size_t k = 0; k < N; ++k) {
      const double gamma = std::atan2(W.at(k, vz), std::fabs(W.at(k, vx)));
      kn[k] = W.at(k, accel) - gravity(W.at(k, altitude)) * std::sin(gamma);
    }
    return integrated_mismatch_last(W, speed, kn, 1, window, smooth);
  }

  double speed_vs_components(const L1Window& W) {
    // rolling_median(np.hypot(vx, vz) - speed, smooth)
    return rolling_median_last(W.n, smooth, [&](size_t k) {
      return std::hypot(W.at(k, vx), W.at(k, vz)) - W.at(k, speed);
    });
  }

  double angle_vs_velocity(const L1Window& W) {
    // diff = wrap_deg(angle - degrees(arctan2(vz, |vx|))); np.where(speed > F9_MIN_SPEED, diff, 0.0); median
    return rolling_median_last(W.n, smooth, [&](size_t k) {
      const double fpa = l1::degrees(std::atan2(W.at(k, vz), std::fabs(W.at(k, vx))));
      const double diff = l1::wrap_deg(W.at(k, angle) - fpa);
      return (W.at(k, speed) > min_speed) ? diff : 0.0;
    });
  }

  double q_vs_density(const L1Window& W) {
    // diff = 100.0 * (q - q_hat) / np.maximum(q_hat, floor); np.where(alt > F9_MIN_Q_ALTITUDE, diff, 0.0); median
    return rolling_median_last(W.n, q_smooth, [&](size_t k) {
      const double q_hat = predicted_q(W, k);
      const double diff = (100.0 * (W.at(k, q) - q_hat)) / np_maximum(q_hat, q_floor);
      return (W.at(k, altitude) > min_q_altitude) ? diff : 0.0;
    });
  }

  double accel_envelope(const L1Window& W) { return std::fabs(W.at(W.n - 1, accel)); }

  double jerk_envelope(const L1Window& W) {
    // acc = rolling_median(accel, 5); lagdiff(acc, 10) / lagdiff(t, 10)
    const size_t N = W.n;
    if (N <= kJerkLag) return kNaN;
    const size_t m = static_cast<size_t>(kJerkSmooth);
    auto acc = [&](size_t j) -> double {
      if (N < m || j + 1 < m) return kNaN;
      double* v = scratch(med, m);
      for (size_t i = 0; i < m; ++i) v[i] = W.at(j + 1 - m + i, accel);
      return np_median(v, m);
    };
    const double a = acc(N - 1);
    const double b = acc(N - 1 - kJerkLag);
    return (a - b) / (W.t[N - 1] - W.t[N - 1 - kJerkLag]);
  }

  double corridor(const L1Window& W) {
    if (!has_cor) return 0.0;   // np.zeros(t.shape)
    const size_t k = W.n - 1;
    const double t = W.t[k];
    double worst = 0.0;
    const struct { size_t ch; const std::vector<double>* lo; const std::vector<double>* hi; double floor; } parts[2] = {
        {altitude, &alt_lo, &alt_hi, alt_floor}, {speed, &spd_lo, &spd_hi, spd_floor}};
    for (const auto& p : parts) {
      // lo = np.where(t <= t_lo_end, np.interp(t, tg, lo), -inf); hi = np.interp(t, tg, hi, right=inf)
      const double lo = (t <= t_lo_end) ? l1::np_interp(t, cor_t, *p.lo) : -kInf;
      const double hi = l1::np_interp(t, cor_t, *p.hi, p.hi->front(), kInf);
      const double scale = np_maximum(l1::np_interp(t, cor_t, *p.hi), p.floor);
      const double x = W.at(k, p.ch);
      const double excess = np_maximum(lo - x, 0.0) + np_maximum(x - hi, 0.0);
      worst = np_maximum(worst, (100.0 * excess) / scale);
    }
    return worst;
  }

  double frozen_channel(const L1Window& W) {
    // np.max(np.where(live, frozen_run_length(X, cap), 0.0)) at the newest frame; live = speed > 5 (and, for q,
    // predicted_q > 50)
    const size_t k = W.n - 1;
    const double* r = frozen_run_last(W, W.n, frozen_cap);
    const bool moving = W.at(k, speed) > kFrozenMinSpeed;
    const bool q_live = predicted_q(W, k) > kFrozenMinQ;
    double best = 0.0;
    for (size_t ch = 0; ch < C; ++ch) {
      const bool live = moving && (ch != q || q_live);
      const double v = live ? r[ch] : 0.0;
      best = (ch == 0) ? v : np_maximum(best, v);
    }
    return best;
  }

  // ---------------------------------------------------------------------------------------- messages

  static std::string msg_rate(const char* what, const char* unit, double value, double threshold) {
    // f"{what} by {abs(value):.0f} {unit} (limit {threshold:.0f} {unit})"
    return std::string(what) + " by " + l1::fmt_fixed(std::fabs(value), 0) + " " + unit + " (limit " +
           l1::fmt_fixed(threshold, 0) + " " + unit + ")";
  }

  std::string msg_q(double value, double threshold, const L1Window& W) const {
    const size_t k = W.n - 1;
    const double qv = W.at(k, q);
    const double q_hat = predicted_q(W, k);
    return "Reported dynamic pressure " + l1::fmt_fixed(qv / 1000.0, 1) +
           " kPa disagrees with \xC2\xBD\xCF\x81v\xC2\xB2 = " + l1::fmt_fixed(q_hat / 1000.0, 1) + " kPa by " +
           l1::fmt_fixed(std::fabs(value), 0) + "% (limit " + l1::fmt_fixed(threshold, 0) + "%)";
  }

  static std::string msg_corridor(double value, double threshold) {
    return "Reported altitude/speed is " + l1::fmt_fixed(value, 0) +
           "% outside the nominal ascent corridor for this mission time (limit " + l1::fmt_fixed(threshold, 0) +
           "%)";
  }

  std::string msg_frozen(double value, double threshold, const L1Window& W) {
    // run = frozen_run_length(X[-int(value) - 2:], int(value) + 1)[-1]; channels with run >= threshold
    const long iv = static_cast<long>(value);   // int(): truncation toward zero
    const size_t n_last = iv + 2 > 0 ? static_cast<size_t>(iv + 2) : W.n;
    const double* r = frozen_run_last(W, n_last, static_cast<double>(iv + 1));
    std::string which;
    for (size_t ch = 0; ch < C; ++ch) {
      if (r[ch] >= threshold) {
        if (!which.empty()) which += ", ";
        which += channels[ch];
      }
    }
    if (which.empty()) which = "a channel";
    return "Channel " + which + " frozen for " + l1::fmt_fixed(value / rate_hz, 1) +
           " s; real sensors never repeat exactly (limit " + l1::fmt_fixed(threshold / rate_hz, 1) + " s)";
  }
};

std::shared_ptr<F9> make_f9(const Config& cfg) {
  auto s = std::make_shared<F9>();
  s->channels = cfg.words("channels");
  s->C = s->channels.size();
  auto idx = [&](const char* name) -> size_t {
    for (size_t i = 0; i < s->channels.size(); ++i) {
      if (s->channels[i] == name) return i;
    }
    throw std::runtime_error(std::string("L1 falcon9 rules need channel '") + name + "' in `channels`");
  };
  s->speed = idx("speed");
  s->altitude = idx("altitude");
  s->vz = idx("vz");
  s->vx = idx("vx");
  s->downrange = idx("downrange");
  s->accel = idx("accel");
  s->angle = idx("angle");
  s->q = idx("q");

  auto cst = [&](const char* name, double fallback) { return cfg.num_or(std::string("l1.const.") + name, fallback); };
  auto ccst = [&](const char* name, double fallback) {
    return as_int(cst(name, fallback), std::string("l1.const.") + name);
  };
  s->rate_hz = cst("F9_RATE_HZ", 10.0);
  s->min_speed = cst("F9_MIN_SPEED", 50.0);
  s->min_flight_speed = cst("F9_MIN_FLIGHT_SPEED", 10.0);
  s->min_q_altitude = cst("F9_MIN_Q_ALTITUDE", 200.0);
  s->g0 = cst("G0", 9.80665);
  s->r_earth = cst("R_EARTH", 6371000.0);
  s->long_window = ccst("F9_LONG_WINDOW", 600);
  const long f9_window = ccst("F9_WINDOW", 100);
  const long f9_smooth = ccst("F9_SMOOTH", 10);

  auto par = [&](const char* name, double fallback) { return cfg.num_or(std::string("l1.params.") + name, fallback); };
  auto ipar = [&](const char* name, double fallback) {
    return as_int(par(name, fallback), std::string("l1.params.") + name);
  };
  s->window = ipar("window", static_cast<double>(f9_window));
  s->smooth = ipar("smooth", static_cast<double>(f9_smooth));
  s->q_smooth = ipar("q_smooth", 5);
  s->q_floor = par("q_floor", 1000.0);
  s->frozen_cap = par("frozen_cap", 50.0);

  auto strictly_increasing = [](const std::vector<double>& v, const char* key) {
    for (size_t i = 1; i < v.size(); ++i) {
      if (!(v[i] > v[i - 1])) throw std::runtime_error(std::string("L1 config: ") + key + " must be strictly increasing");
    }
  };
  if (has_prefix(cfg, "l1.params.atmosphere.")) {   // `if atm:` (a non-empty dict)
    s->has_atm = true;
    s->atm_alt = cfg.vec("l1.params.atmosphere.alt");
    s->atm_log_rho = cfg.vec("l1.params.atmosphere.log_rho");
    if (s->atm_alt.empty() || s->atm_alt.size() != s->atm_log_rho.size()) {
      throw std::runtime_error("L1 config: atmosphere.alt / atmosphere.log_rho must be non-empty and equal length");
    }
    strictly_increasing(s->atm_alt, "l1.params.atmosphere.alt");
  }
  if (has_prefix(cfg, "l1.params.corridor.")) {   // `if not cor: return zeros`
    s->has_cor = true;
    s->cor_t = cfg.vec("l1.params.corridor.t");
    s->alt_lo = cfg.vec("l1.params.corridor.altitude_lo");
    s->alt_hi = cfg.vec("l1.params.corridor.altitude_hi");
    s->spd_lo = cfg.vec("l1.params.corridor.speed_lo");
    s->spd_hi = cfg.vec("l1.params.corridor.speed_hi");
    s->alt_floor = cfg.num("l1.params.corridor.altitude_floor");
    s->spd_floor = cfg.num("l1.params.corridor.speed_floor");
    const size_t n = s->cor_t.size();
    for (const auto* v : {&s->alt_lo, &s->alt_hi, &s->spd_lo, &s->spd_hi}) {
      if (n == 0 || v->size() != n) throw std::runtime_error("L1 config: corridor tables must match corridor.t");
    }
    strictly_increasing(s->cor_t, "l1.params.corridor.t");
    s->t_lo_end = cfg.num_or("l1.params.corridor.t_all_end", s->cor_t.back());   // cor.get("t_all_end", tg[-1])
  }
  s->scratch(s->med, static_cast<size_t>(std::max<long>({s->smooth, s->q_smooth, kJerkSmooth, 1})));
  s->scratch(s->run, s->C);
  return s;
}

// Rule table: name -> (residual, message, default lookback, default threshold), as FALCON9_RULES.
L1Rule f9_rule(const std::string& name, const std::shared_ptr<F9>& s, long f9_window, long f9_smooth) {
  using Res = double (F9::*)(const L1Window&);
  L1Rule r;
  r.name = name;
  auto bind = [&](Res fn) { r.residual = [s, fn](const L1Window& w) { return ((*s).*fn)(w); }; };
  auto rate = [&](const char* what, const char* unit) {
    r.message = [what, unit](double v, double thr, const L1Window&) { return F9::msg_rate(what, unit, v, thr); };
  };
  auto set = [&](long lookback, double threshold) {
    r.lookback = static_cast<size_t>(lookback);
    r.threshold = threshold;
  };
  if (name == "vz_vs_altitude_rate") {
    bind(&F9::vz_vs_altitude);
    rate("Reported vertical velocity disagrees with altitude rate", "m/s");
    set(f9_window + f9_smooth, 90.0);
  } else if (name == "vz_vs_altitude_rate_60s") {
    bind(&F9::vz_vs_altitude_long);
    rate("Over the last minute, altitude has drifted from integrated vertical velocity", "m/s");
    set(s->long_window + f9_smooth, 25.0);
  } else if (name == "vx_vs_downrange_rate") {
    bind(&F9::vx_vs_downrange);
    rate("Reported horizontal velocity disagrees with downrange rate", "m/s");
    set(f9_window + f9_smooth, 2.0);
  } else if (name == "accel_vs_speed_rate") {
    bind(&F9::accel_vs_speed);
    rate("Reported acceleration (minus gravity) disagrees with speed rate", "m/s\xC2\xB2");
    set(f9_window + f9_smooth, 12.0);
  } else if (name == "speed_vs_components") {
    bind(&F9::speed_vs_components);
    rate("Reported speed disagrees with \xE2\x88\x9A(vx\xC2\xB2 + vz\xC2\xB2)", "m/s");
    set(f9_smooth, 45.0);
  } else if (name == "angle_vs_velocity") {
    bind(&F9::angle_vs_velocity);
    rate("Reported flight-path angle disagrees with the velocity vector", "deg");
    set(f9_smooth, 12.0);
  } else if (name == "q_vs_density") {
    bind(&F9::q_vs_density);
    r.message = [s](double v, double thr, const L1Window& w) { return s->msg_q(v, thr, w); };
    set(5, 5.0);
  } else if (name == "accel_envelope") {
    bind(&F9::accel_envelope);
    rate("Reported acceleration is beyond what Falcon 9 can do", "m/s\xC2\xB2");
    set(0, 60.0);
  } else if (name == "jerk_envelope") {
    bind(&F9::jerk_envelope);
    rate("Reported acceleration changes faster than the engines can throttle", "m/s\xC2\xB3");
    set(15, 25.0);
  } else if (name == "corridor") {
    bind(&F9::corridor);
    r.message = [](double v, double thr, const L1Window&) { return F9::msg_corridor(v, thr); };
    set(0, 25.0);
  } else if (name == "frozen_channel") {
    bind(&F9::frozen_channel);
    r.message = [s](double v, double thr, const L1Window& w) { return s->msg_frozen(v, thr, w); };
    set(51, 20.0);
  } else {
    throw std::runtime_error("unknown L1 rule '" + name + "' for schema falcon9");
  }
  return r;
}

}  // namespace

// ============================================================================================ rule registry

std::vector<L1Rule> make_l1_rules(const Config& cfg) {
  const std::string schema = cfg.words("schema").at(0);
  std::vector<std::string> names;
  if (cfg.strs.count("l1.rules")) {
    names = cfg.strs.at("l1.rules");
  } else if (!cfg.has("l1.rules")) {
    throw std::runtime_error("L1 config: missing l1.rules");
  }   // else: `l1.rules` with no values -> no rules

  std::vector<L1Rule> rules;
  if (schema == "falcon9") {
    auto s = make_f9(cfg);
    const long f9_window = as_int(cfg.num_or("l1.const.F9_WINDOW", 100), "l1.const.F9_WINDOW");
    const long f9_smooth = as_int(cfg.num_or("l1.const.F9_SMOOTH", 10), "l1.const.F9_SMOOTH");
    for (const auto& n : names) rules.push_back(f9_rule(n, s, f9_window, f9_smooth));
  } else {
    // Add a context + rule table like F9 above for other schemas (quad: pos_vs_velocity, velocity_vs_accel, ...).
    throw std::runtime_error("L1 rules for schema '" + schema + "' are not implemented in vbw_edge (falcon9 only)");
  }

  for (auto& r : rules) {
    const std::string lk = "l1.lookback." + r.name, tk = "l1.threshold." + r.name;
    const long lb = as_int(cfg.num_or(lk, static_cast<double>(r.lookback)), lk);
    if (lb < 0) throw std::runtime_error("L1 config: " + lk + " must be >= 0");
    r.lookback = static_cast<size_t>(lb);
    r.threshold = cfg.num_or(tk, r.threshold);
    if (!(r.threshold > 0) || !std::isfinite(r.threshold)) throw std::runtime_error("L1 config: " + tk + " must be > 0");
  }
  return rules;
}

// ============================================================================================ L1Checker

namespace {

size_t buffer_capacity(const std::vector<L1Rule>& rules, const Config& cfg) {
  size_t lb = 0;
  for (const auto& r : rules) lb = std::max(lb, r.lookback);
  size_t cap = lb + 1;   // FrameBuffer(len(channels), max_lookback + 1)
  // The L2 model reads its window (l2.window + 2 frames) from this buffer too; never make it shorter than that.
  if (cfg.has("l2.window")) cap = std::max(cap, static_cast<size_t>(as_int(cfg.num("l2.window"), "l2.window")) + 2);
  return cap;
}

}  // namespace

L1Checker::L1Checker(const Config& cfg)
    : rules_(make_l1_rules(cfg)), buffer_(cfg.words("channels").size(), buffer_capacity(rules_, cfg)) {
  for (const auto& r : rules_) {
    names_.push_back(r.name);
    thresholds_.push_back(r.threshold);
    lookbacks_.push_back(r.lookback);
  }
  cands_.reserve(rules_.size());
}

void L1Checker::reset() { buffer_.clear(); }

bool L1Checker::window_for(size_t i, L1Window& w) const {
  const size_t n = std::min(rules_.at(i).lookback + 1, buffer_.size());
  w = L1Window{buffer_.last_X(n), buffer_.last_t(n), n, buffer_.channels()};
  return buffer_.size() > rules_[i].lookback;   // check_rule: X.shape[0] <= lookback -> not evaluable
}

double L1Checker::residual(size_t i) const {
  L1Window w{};
  if (!window_for(i, w)) return kNaN;
  return rules_[i].residual(w);
}

std::string L1Checker::message(size_t i, double value) const {
  L1Window w{};
  window_for(i, w);
  if (w.n == 0) return "";
  return rules_[i].message(value, rules_[i].threshold, w);
}

L1Result L1Checker::update(const std::vector<double>& x, double t) {
  L1Result out;
  update(x, t, out);
  return out;
}

void L1Checker::update(const std::vector<double>& x, double t, L1Result& out) {
  buffer_.append(x, t);
  const size_t R = rules_.size();
  out.ratios.resize(R);
  out.max_ratio = 0.0;
  out.violated.clear();
  out.message.clear();
  cands_.clear();
  for (size_t i = 0; i < R; ++i) {
    const L1Rule& rule = rules_[i];
    double ratio = 0.0;
    L1Window w{};
    if (window_for(i, w)) {
      const double value = rule.residual(w);   // float(rule.residual(X, t, params)[-1])
      if (std::isfinite(value)) {
        ratio = value / rule.threshold;
        if (!(std::fabs(value) <= rule.threshold)) {
          // Violation.value = round(value, 4); sorted by -abs(v.value) / v.threshold (stable)
          cands_.push_back({static_cast<int>(i), (-std::fabs(l1::py_round(value, 4))) / rule.threshold, value});
        }
      }
    }
    out.ratios[i] = ratio;
    out.max_ratio = std::max(out.max_ratio, std::fabs(ratio));
  }
  if (cands_.empty()) return;
  std::stable_sort(cands_.begin(), cands_.end(), [](const Cand& a, const Cand& b) { return a.key < b.key; });
  for (const auto& c : cands_) out.violated.push_back(c.rule);
  const Cand& worst = cands_.front();
  L1Window w{};
  window_for(static_cast<size_t>(worst.rule), w);
  out.message = rules_[static_cast<size_t>(worst.rule)].message(worst.value, rules_[static_cast<size_t>(worst.rule)].threshold, w);
}

}  // namespace vbw

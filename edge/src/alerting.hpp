// Score combination and alert persistence / hysteresis (mirrors vbw/detector/alerting.py).
//
// An alert rises when the score exceeds tau on at least k of the last n frames, and clears only after
// clear_frames consecutive frames below clear_ratio * tau. The first `warmup` frames after reset never alert.
#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace vbw {

struct Config;

// Map a ratio-to-threshold into 0..1, exactly 0.5 at the threshold: r^2 / (1 + r^2).
double soft_ratio(double r);

// Noisy-OR of scores, each clipped to [0, 1] (NaN propagates, as in Python's min/max).
double combine(std::initializer_list<double> scores);

struct AlertConfig {
  double tau = 0.5;          // score needed to count a frame as suspicious
  int k = 3;                 // suspicious frames needed ...
  int n = 5;                 // ... within the last n frames to raise
  int clear_frames = 20;     // consecutive quiet frames needed to clear
  double clear_ratio = 0.8;  // quiet means score < clear_ratio * tau
  int warmup = 10;           // frames after reset that can never alert

  // alert.tau, alert.k, alert.n, alert.clear_frames, alert.clear_ratio, alert.warmup (defaults when missing,
  // as AlertConfig.load falls back to AlertConfig()).
  static AlertConfig from(const Config& cfg);
};

// k-of-n raise, m-quiet-frames clear. Pure state machine over a stream of scores.
class AlertState {
 public:
  explicit AlertState(AlertConfig cfg);
  void reset();
  bool update(double score);
  bool active() const { return active_; }
  const AlertConfig& config() const { return cfg_; }

 private:
  AlertConfig cfg_;
  std::vector<unsigned char> ring_;  // deque(maxlen=n) of "score > tau"
  std::size_t head_ = 0;             // index of the oldest entry
  std::size_t count_ = 0;            // entries held (<= n)
  int suspicious_ = 0;               // sum(recent)
  int quiet_ = 0;
  long long frames_ = 0;
  bool active_ = false;
};

}  // namespace vbw

#include "alerting.hpp"

#include "config.hpp"

namespace vbw {

double soft_ratio(double r) {
  const double r2 = r * r;
  return r2 / (1.0 + r2);
}

double combine(std::initializer_list<double> scores) {
  double keep = 1.0;
  for (double s : scores) {
    // min(max(s, 0.0), 1.0) with Python's argument order, so NaN propagates exactly as in the reference.
    double v = s;
    if (0.0 > v) v = 0.0;
    if (1.0 < v) v = 1.0;
    keep *= 1.0 - v;
  }
  return 1.0 - keep;
}

AlertConfig AlertConfig::from(const Config& cfg) {
  AlertConfig a;
  a.tau = cfg.num_or("alert.tau", a.tau);
  a.k = static_cast<int>(cfg.num_or("alert.k", a.k));
  a.n = static_cast<int>(cfg.num_or("alert.n", a.n));
  a.clear_frames = static_cast<int>(cfg.num_or("alert.clear_frames", a.clear_frames));
  a.clear_ratio = cfg.num_or("alert.clear_ratio", a.clear_ratio);
  a.warmup = static_cast<int>(cfg.num_or("alert.warmup", a.warmup));
  return a;
}

AlertState::AlertState(AlertConfig cfg) : cfg_(cfg) {
  ring_.assign(cfg_.n > 0 ? static_cast<std::size_t>(cfg_.n) : 0, 0);
  reset();
}

void AlertState::reset() {
  head_ = 0;
  count_ = 0;
  suspicious_ = 0;
  quiet_ = 0;
  frames_ = 0;
  active_ = false;
}

bool AlertState::update(double score) {
  frames_ += 1;
  if (frames_ <= cfg_.warmup) return false;

  // self._recent.append(score > tau) on a deque(maxlen=n)
  const unsigned char hit = score > cfg_.tau ? 1 : 0;
  const std::size_t cap = ring_.size();
  if (cap > 0) {
    if (count_ == cap) {  // full: drop the oldest
      suspicious_ -= ring_[head_];
      ring_[head_] = hit;
      head_ = (head_ + 1) % cap;
    } else {
      ring_[(head_ + count_) % cap] = hit;
      ++count_;
    }
    suspicious_ += hit;
  }

  if (!active_) {
    if (suspicious_ >= cfg_.k) {
      active_ = true;
      quiet_ = 0;
    }
  } else {
    quiet_ = score < cfg_.clear_ratio * cfg_.tau ? quiet_ + 1 : 0;
    if (quiet_ >= cfg_.clear_frames) {
      active_ = false;
      head_ = 0;  // self._recent.clear()
      count_ = 0;
      suspicious_ = 0;
    }
  }
  return active_;
}

}  // namespace vbw

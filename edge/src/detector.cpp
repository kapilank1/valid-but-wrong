#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

#include "config.hpp"

namespace vbw {

namespace {

// ", ".join(l0.reasons or ["non-finite values"]). The C++ L0 carries flags only, so the reasons are the flag
// names without the Python's numeric details (sync word, seq / time values).
std::string link_reasons(const L0& l0) {
  std::string out;
  auto add = [&out](const char* r) {
    if (!out.empty()) out += ", ";
    out += r;
  };
  if (!l0.sync_ok) add("bad sync");
  if (!l0.crc_ok) add("CRC mismatch");
  if (!l0.seq_ok) add("seq jump");
  if (!l0.time_ok) add("time not monotonic");
  if (out.empty()) out = "non-finite values";
  return out;
}

}  // namespace

std::string l2_explanation(const std::vector<double>& z, const std::vector<std::string>& channels) {
  // i = int(np.argmax(np.abs(z))): first maximum; a NaN wins at its first occurrence.
  size_t i = 0;
  double best = z.empty() ? 0.0 : std::fabs(z[0]);
  if (!std::isnan(best)) {
    for (size_t k = 1; k < z.size(); ++k) {
      const double a = std::fabs(z[k]);
      if (std::isnan(a)) {
        i = k;
        break;
      }
      if (a > best) {
        best = a;
        i = k;
      }
    }
  }
  const double v = z.empty() ? 0.0 : std::fabs(z[i]);
  char num[64];
  std::snprintf(num, sizeof num, "%.1f", v);
  const std::string name = i < channels.size() ? channels[i] : std::string("channel ") + std::to_string(i);
  return name + " deviates from the learned flight dynamics (" + num + " sigma from the predicted next value)";
}

BaseDetector::BaseDetector(const Config& cfg)
    : channels_(cfg.words("channels")),
      l1_(cfg),
      l2_(l2_available(cfg) ? std::make_unique<L2Model>(cfg) : nullptr),
      alert_cfg_(AlertConfig::from(cfg)),
      alert_(alert_cfg_) {
  if (l2_ && l2_->channels() != channels_.size())
    throw std::runtime_error("detector: L2 norm has " + std::to_string(l2_->channels()) + " channels, config has " +
                             std::to_string(channels_.size()));
  const size_t C = channels_.size();
  x_.reserve(C);
  z_.assign(C, 0.0);
  if (l2_) {
    X_.reserve(static_cast<size_t>(l2_->needs()) * C);
    tbuf_.reserve(static_cast<size_t>(l2_->needs()));
  }
  reset();
}

BaseDetector::~BaseDetector() = default;

void BaseDetector::reset() {
  const size_t C = channels_.size();
  l1_.reset();
  alert_.reset();
  ewma_.assign(C, 0.0);
  has_last_t_ = false;
  last_t_ = 0;
  reason_.clear();
  last_l1_ = L1Result{};
  last_z_.assign(C, 0.0);
  last_accepted_ = false;
}

// _l2_step: z-scores of the newest frame (into z_) and the L2 ratio (0 while the history is too short).
double BaseDetector::l2_step() {
  std::fill(z_.begin(), z_.end(), 0.0);
  if (!l2_ || l1_.buffer().size() < static_cast<size_t>(l2_->needs())) return 0.0;
  l1_.buffer().last(static_cast<size_t>(l2_->needs()), X_, tbuf_);
  l2_->zscore_last(X_.data(), z_.data());
  ewma_update(ewma_, z_, l2_->norm().ewma_alpha);
  return l2_ratio(z_, ewma_, l2_->norm());
}

std::string BaseDetector::explain(const L1Result& l1, double l1_score, double l2_score) const {
  if (!l1.violated.empty() && l1_score >= l2_score) return l1.message;
  if (l2_score > 0) {
    const bool any = std::any_of(z_.begin(), z_.end(), [](double v) { return v != 0.0; });  // np.any (NaN is true)
    return l2_explanation(any ? z_ : ewma_, channels_);
  }
  return !l1.violated.empty() ? l1.message : std::string();
}

Verdict BaseDetector::score_frame(const Frame& f, const L0& l0) {
  Verdict v;
  v.t = f.t;
  v.seq = f.seq;
  v.l0 = l0;
  last_accepted_ = false;

  // frame_vector: wire-order values, NaN for a missing channel.
  const size_t C = channels_.size();
  x_.assign(C, std::numeric_limits<double>::quiet_NaN());
  std::copy_n(f.values.begin(), std::min(C, f.values.size()), x_.begin());
  const bool finite = std::all_of(x_.begin(), x_.end(), [](double x) { return std::isfinite(x); });

  if (!(l0.sync_ok && l0.crc_ok) || !finite) {
    // Corrupted bits are L0's job; don't let them poison the physics history.
    v.explanation = "Link integrity failure: " + link_reasons(l0);
  } else if (has_last_t_ && f.t <= last_t_) {
    v.explanation = "Out-of-order frame skipped by physics checks";
  } else {
    last_t_ = f.t;
    has_last_t_ = true;
    last_accepted_ = true;

    L1Result l1 = l1_.update(x_, f.t);
    const double ratio2 = l2_step();
    v.accepted = true;
    v.l1_ratios = l1.ratios;
    v.l2_z = z_;
    v.l1_score = soft_ratio(l1.max_ratio);
    v.l2_score = soft_ratio(ratio2);
    v.score = combine({v.l1_score, v.l2_score});
    if (v.score > alert_cfg_.tau) reason_ = explain(l1, v.l1_score, v.l2_score);
    last_l1_ = std::move(l1);
    last_z_ = z_;
    return v;
  }
  v.l1_ratios.assign(l1_.rule_names().size(), 0.0);
  v.l2_z.assign(C, 0.0);
  return v;
}

Verdict BaseDetector::update(const Frame& f, const L0& l0) {
  Verdict v = score_frame(f, l0);
  v.alert = alert_.update(v.score);  // every frame, rejected ones included (score 0), as in Python
  if (v.alert)
    v.explanation = reason_;
  else if (v.score > alert_cfg_.tau && v.explanation.empty())
    v.explanation = reason_;
  return v;
}

}  // namespace vbw

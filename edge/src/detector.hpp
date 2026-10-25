// BaseDetector: L1 physics rules + L2 learned predictor, combined into one score with persistent alerting.
// Line-by-line port of vbw/detector/base.py (BaseDetector._score / update / _l2_step / _explain).
//
// l1_score = soft(max rule residual / threshold); l2_score = soft(max(instant |z|, |EWMA z|) / thresholds);
// score = noisy-OR of the two; alert = k-of-n persistence with hysteresis (alerting.hpp).
// Without an L2 model file (cfg l2.onnx missing) the detector runs L1-only (l2_score = 0), like Python.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "alerting.hpp"
#include "frame.hpp"
#include "l1.hpp"
#include "l2.hpp"

namespace vbw {

struct Config;

struct Verdict {
  double t = 0;
  uint16_t seq = 0;
  bool accepted = false;                 // frame entered the physics history (BaseDetector.last_accepted)
  L0 l0;
  std::vector<double> l1_ratios, l2_z;   // rule order / channel order; zeros when the frame was not accepted
  double l1_score = 0, l2_score = 0, score = 0;
  bool alert = false;
  std::string explanation;
};

// "<channel> deviates from the learned flight dynamics (<|z|:.1f> sigma from the predicted next value)"
std::string l2_explanation(const std::vector<double>& z, const std::vector<std::string>& channels);

class BaseDetector {
 public:
  explicit BaseDetector(const Config& cfg);
  ~BaseDetector();
  BaseDetector(const BaseDetector&) = delete;
  BaseDetector& operator=(const BaseDetector&) = delete;

  void reset();
  Verdict update(const Frame& f, const L0& l0);   // mirrors BaseDetector.update in vbw/detector/base.py

  // Python attributes, for tests and tools.
  const std::vector<std::string>& channels() const { return channels_; }
  const std::vector<std::string>& rule_names() const { return l1_.rule_names(); }
  const AlertConfig& alert_config() const { return alert_cfg_; }
  bool has_l2() const { return l2_ != nullptr; }
  const L2Model* l2() const { return l2_.get(); }
  const L1Checker& l1() const { return l1_; }
  bool last_accepted() const { return last_accepted_; }
  const L1Result& last_l1() const { return last_l1_; }     // ratios empty until the first accepted frame
  const std::vector<double>& last_z() const { return last_z_; }
  const std::vector<double>& ewma() const { return ewma_; }
  bool alert_active() const { return alert_.active(); }

 private:
  Verdict score_frame(const Frame& f, const L0& l0);                 // Python _score
  double l2_step();                                                   // writes z_, returns the L2 ratio
  std::string explain(const L1Result& l1, double l1_score, double l2_score) const;

  std::vector<std::string> channels_;
  L1Checker l1_;
  std::unique_ptr<L2Model> l2_;
  AlertConfig alert_cfg_;
  AlertState alert_;

  // state (reset())
  std::vector<double> ewma_;
  bool has_last_t_ = false;
  double last_t_ = 0;
  std::string reason_;
  L1Result last_l1_;
  std::vector<double> last_z_;
  bool last_accepted_ = false;

  // reusable scratch (no per-frame allocation once warm)
  std::vector<double> x_, X_, tbuf_, z_;
};

}  // namespace vbw

// BaseDetector parity vs the Python reference (edge/testdata/golden_*.csv, written by scripts/edge_golden.py), plus
// unit tests for soft_ratio / combine / AlertState mirroring tests/test_detector_alerting.py.
//
//   test_detector [--edge-dir DIR] [--skip-l1] [golden.csv ...]
//
// Without files, every edge/testdata/golden_*.csv is checked. Each file's schema config (edge/config/*.cfg) is the
// one whose channels match the file's l2z.* columns. Frames come from the packet_hex column (decode + LinkMonitor,
// so L0 is exercised) when present, else from t / seq / channel values. Exit code is non-zero on any failure.
// --skip-l1 compares only L1-independent columns (accepted, l2z, l2_score); for bring-up with a stand-in L1.

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS  // std::getenv
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "alerting.hpp"
#include "config.hpp"
#include "detector.hpp"
#include "frame.hpp"

namespace fs = std::filesystem;
using namespace vbw;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

// --------------------------------------------------------------------------- unit tests (alerting)

AlertConfig cfg_kw(int warmup = 0, double tau = 0.5) {
  AlertConfig c;
  c.tau = tau;
  c.k = 3;
  c.n = 5;
  c.clear_frames = 4;
  c.clear_ratio = 0.8;
  c.warmup = warmup;
  return c;
}

std::vector<bool> alert_series(const std::vector<double>& scores, const AlertConfig& c) {
  AlertState s(c);
  std::vector<bool> out;
  for (double x : scores) out.push_back(s.update(x));
  return out;
}

void unit_tests() {
  // soft_ratio is half at the threshold and monotonic
  CHECK(soft_ratio(1.0) == 0.5);
  CHECK(soft_ratio(0.0) == 0.0);
  CHECK(soft_ratio(0.5) < soft_ratio(1.0) && soft_ratio(1.0) < soft_ratio(2.0) && soft_ratio(2.0) < 1.0);
  CHECK(soft_ratio(-2.0) == soft_ratio(2.0));

  // combine is noisy-OR, clipped
  CHECK(combine({0.0, 0.0}) == 0.0);
  CHECK(combine({0.5, 0.5}) == 0.75);
  CHECK(combine({1.0, 0.2}) == 1.0);
  CHECK(combine({-0.3, 0.4}) == combine({0.0, 0.4}));
  CHECK(combine({1.7, 0.0}) == 1.0);
  CHECK(std::isnan(combine({std::nan(""), 0.1})));  // Python min(max(nan, 0), 1) is nan

  // a single spike never alerts
  {
    std::vector<double> s(10, 0.1);
    s.push_back(0.99);
    s.insert(s.end(), 10, 0.1);
    auto out = alert_series(s, cfg_kw());
    CHECK(std::none_of(out.begin(), out.end(), [](bool b) { return b; }));
  }
  // k-of-n raises on the third suspicious frame
  {
    auto out = alert_series({0.1, 0.9, 0.1, 0.9, 0.9, 0.1}, cfg_kw());
    CHECK((out == std::vector<bool>{false, false, false, false, true, true}));
  }
  // hysteresis holds through borderline scores and clears after quiet frames
  {
    std::vector<double> s(3, 0.9);
    s.insert(s.end(), 6, 0.45);
    s.insert(s.end(), 5, 0.1);
    auto out = alert_series(s, cfg_kw());
    CHECK(out[2]);
    CHECK(std::all_of(out.begin() + 3, out.begin() + 9, [](bool b) { return b; }));
    CHECK(out[9] && out[10] && out[11]);
    CHECK(!out[12]);
    CHECK(!out[13]);
  }
  // after clearing, the k-of-n window starts empty again
  {
    std::vector<double> s(3, 0.9);
    s.insert(s.end(), 4, 0.1);   // clears on the 4th quiet frame
    s.push_back(0.9);
    s.push_back(0.9);
    s.push_back(0.9);
    auto out = alert_series(s, cfg_kw());
    CHECK(out[2] && !out[6]);
    CHECK(!out[7] && !out[8] && out[9]);
  }
  // warm-up suppresses alerts
  {
    auto out = alert_series(std::vector<double>(15, 1.0), cfg_kw(10));
    CHECK(std::none_of(out.begin(), out.begin() + 12, [](bool b) { return b; }));
    CHECK(std::all_of(out.begin() + 12, out.end(), [](bool b) { return b; }));
  }
  // reset restarts warm-up and history
  {
    AlertState s(cfg_kw(2));
    for (int i = 0; i < 6; ++i) s.update(1.0);
    CHECK(s.active());
    s.reset();
    CHECK(!s.active());
    CHECK(!s.update(1.0) && !s.update(1.0) && !s.update(1.0) && !s.update(1.0) && s.update(1.0));
  }
  // score == tau is not suspicious (strict >), score == clear level is not quiet (strict <)
  {
    auto out = alert_series({0.5, 0.5, 0.5, 0.5, 0.5}, cfg_kw());
    CHECK(std::none_of(out.begin(), out.end(), [](bool b) { return b; }));
  }
  // l2 explanation text and Python-style rounding (3.25 -> "3.2", half to even on the exact binary value)
  {
    const std::vector<std::string> ch{"speed", "altitude", "vz"};
    CHECK(l2_explanation({0.0, -3.25, 1.0}, ch) ==
          "altitude deviates from the learned flight dynamics (3.2 sigma from the predicted next value)");
    CHECK(l2_explanation({4.0, -4.0, 1.0}, ch) ==
          "speed deviates from the learned flight dynamics (4.0 sigma from the predicted next value)");
    CHECK(l2_explanation({0.0, 0.0, 12.345}, ch) ==
          "vz deviates from the learned flight dynamics (12.3 sigma from the predicted next value)");
  }
}

// --------------------------------------------------------------------------- golden CSV

struct Csv {
  std::vector<std::string> header;
  std::map<std::string, size_t> col;
  std::vector<std::vector<std::string>> rows;
  bool has(const std::string& c) const { return col.count(c) > 0; }
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : line) {
    if (ch == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (ch != '\r' && ch != '\n') {
      cur += ch;
    }
  }
  out.push_back(cur);
  return out;
}

Csv read_csv(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + p.string());
  Csv csv;
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty csv " + p.string());
  if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) line = line.substr(3);
  csv.header = split(line);
  for (size_t i = 0; i < csv.header.size(); ++i) csv.col[csv.header[i]] = i;
  while (std::getline(in, line)) {
    if (line.empty() || line == "\r") continue;
    auto r = split(line);
    if (r.size() != csv.header.size())
      throw std::runtime_error(p.string() + ": row " + std::to_string(csv.rows.size() + 2) + " has " +
                               std::to_string(r.size()) + " fields, header has " + std::to_string(csv.header.size()));
    csv.rows.push_back(std::move(r));
  }
  return csv;
}

double to_num(const std::string& s) {
  if (s == "True" || s == "true") return 1;
  if (s == "False" || s == "false") return 0;
  if (s.empty()) return std::nan("");
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str()) throw std::runtime_error("not a number: '" + s + "'");
  return v;
}

bool to_bool(const std::string& s) { return to_num(s) != 0; }

std::vector<uint8_t> from_hex(const std::string& h) {
  std::vector<uint8_t> out(h.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(std::stoul(h.substr(2 * i, 2), nullptr, 16));
  return out;
}

fs::path find_edge_dir(const std::string& hint, const char* argv0) {
  std::vector<fs::path> starts;
  if (!hint.empty()) return fs::path(hint);
  if (const char* env = std::getenv("VBW_EDGE_DIR")) return fs::path(env);
  starts.push_back(fs::current_path());
  std::error_code ec;
  fs::path exe = fs::absolute(fs::path(argv0), ec);
  if (!ec) starts.push_back(exe.parent_path());
  for (fs::path p : starts) {
    for (int up = 0; up < 8 && !p.empty(); ++up) {
      if (fs::exists(p / "config" / "falcon9.cfg") && fs::exists(p / "SPEC.md")) return p;
      if (fs::exists(p / "edge" / "config" / "falcon9.cfg")) return p / "edge";
      if (p == p.parent_path()) break;
      p = p.parent_path();
    }
  }
  throw std::runtime_error("cannot find the edge directory (pass --edge-dir or set VBW_EDGE_DIR)");
}

// The schema config whose channel list matches the golden file's l2z.* columns.
fs::path config_for(const Csv& csv, const fs::path& edge_dir) {
  for (const auto& entry : fs::directory_iterator(edge_dir / "config")) {
    if (entry.path().extension() != ".cfg") continue;
    Config cfg = Config::load(entry.path().string());
    if (!cfg.has("channels")) continue;
    const auto& ch = cfg.words("channels");
    size_t found = 0;
    for (const auto& c : ch) found += csv.has("l2z." + c);
    size_t l2cols = 0;
    for (const auto& h : csv.header) l2cols += h.rfind("l2z.", 0) == 0;
    if (found == ch.size() && l2cols == ch.size()) return entry.path();
  }
  throw std::runtime_error("no config in " + (edge_dir / "config").string() + " matches the golden channels");
}

struct Stat {
  double max_err = 0;
  double at_t = 0;
  long long bad = 0;
  void add(double err, double t, double tol) {
    if (std::isnan(err)) err = INFINITY;
    if (err > max_err) {
      max_err = err;
      at_t = t;
    }
    if (err > tol) ++bad;
  }
};

// |a - b| with nan == nan treated as equal.
double diff(double a, double b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (a == b) return 0;  // equal infinities
  return std::fabs(a - b);
}

// L1 ratio tolerance: abs 1e-9, relative 1e-9 for large values (SPEC section 3).
double l1_err(double a, double b) {
  const double d = diff(a, b);
  const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
  return d / scale;
}

bool check_golden(const fs::path& file, const fs::path& edge_dir, bool skip_l1) {
  const Csv csv = read_csv(file);
  const fs::path cfg_path = config_for(csv, edge_dir);
  const Config cfg = Config::load(cfg_path.string());
  BaseDetector det(cfg);
  if (!det.has_l2()) {
    std::printf("%s: FAIL (no L2 model for %s)\n", file.filename().string().c_str(), cfg_path.string().c_str());
    return false;
  }
  const auto& ch = det.channels();
  const auto& rules = det.rule_names();
  const bool use_packets = csv.has("packet_hex");
  const bool has_l0 = csv.has("l0_ok");
  std::vector<size_t> ch_col, z_col, r_col;
  for (const auto& c : ch) {
    ch_col.push_back(csv.col.at(c));
    z_col.push_back(csv.col.at("l2z." + c));
  }
  bool have_l1r = !skip_l1;
  for (const auto& r : rules) {
    auto it = csv.col.find("l1r." + r);
    if (it == csv.col.end()) {
      have_l1r = false;
      break;
    }
    r_col.push_back(it->second);
  }
  const size_t c_t = csv.col.at("t"), c_seq = csv.col.at("seq"), c_acc = csv.col.at("accepted");
  const size_t c_l1 = csv.col.at("l1_score"), c_l2 = csv.col.at("l2_score"), c_s = csv.col.at("score");
  const size_t c_alert = csv.col.at("alert");

  LinkMonitor mon;
  Frame f;
  Stat s_z, s_l1, s_l2, s_s, s_r;
  long long acc_bad = 0, l0_bad = 0, alert_bad = 0, decode_bad = 0, alerts = 0, golden_alerts = 0;
  std::vector<std::string> notes;
  auto note = [&notes](const std::string& s) {
    if (notes.size() < 12) notes.push_back(s);
  };

  for (const auto& row : csv.rows) {
    const double t_row = to_num(row[c_t]);
    L0 l0;
    if (use_packets) {
      const auto pkt = from_hex(row[csv.col.at("packet_hex")]);
      std::string why;
      if (!decode(pkt.data(), pkt.size(), f, l0, &why)) {
        ++decode_bad;
        note("decode failed at t=" + row[c_t] + ": " + why);
        continue;
      }
      l0 = mon.receive(f, l0);
    } else {
      f = Frame{};
      f.schema_id = static_cast<uint8_t>(cfg.num_or("schema_id", 0));
      f.seq = static_cast<uint16_t>(to_num(row[c_seq]));
      f.t = t_row;
      for (size_t i = 0; i < ch.size(); ++i) f.values.push_back(to_num(row[ch_col[i]]));
      l0 = mon.receive(f, L0{});
      // Without the packet the only other L0 failure is integrity (sync / CRC): reject like Python did.
      if (has_l0 && !to_bool(row[csv.col.at("l0_ok")]) && l0.ok()) l0.crc_ok = false;
    }
    if (has_l0 && l0.ok() != to_bool(row[csv.col.at("l0_ok")])) {
      ++l0_bad;
      note("l0_ok mismatch at t=" + row[c_t]);
    }

    const Verdict v = det.update(f, l0);
    const bool g_acc = to_bool(row[c_acc]);
    if (v.accepted != g_acc || det.last_accepted() != g_acc) {
      ++acc_bad;
      note("accepted mismatch at t=" + row[c_t]);
    }
    // Golden l2z / l1r: BaseDetector.last_z / last_l1 ratios on accepted frames, 0 on rejected ones (the verdict's
    // zeros). Python keeps last_z across a rejected frame, so check that too.
    if (v.accepted && det.last_z() != v.l2_z) {
      ++acc_bad;
      note("last_z differs from verdict z at t=" + row[c_t]);
    }
    for (size_t i = 0; i < ch.size(); ++i) s_z.add(diff(v.l2_z[i], to_num(row[z_col[i]])), t_row, 1e-4);
    if (have_l1r) {
      for (size_t i = 0; i < rules.size(); ++i) s_r.add(l1_err(v.l1_ratios[i], to_num(row[r_col[i]])), t_row, 1e-9);
    }
    for (const auto& [name, flag] : {std::pair<const char*, bool>{"sync_ok", l0.sync_ok}, {"crc_ok", l0.crc_ok},
                                     {"seq_ok", l0.seq_ok}, {"time_ok", l0.time_ok}}) {
      auto it = csv.col.find(name);
      if (it != csv.col.end() && to_bool(row[it->second]) != flag) {
        ++l0_bad;
        note(std::string(name) + " mismatch at t=" + row[c_t]);
      }
    }
    s_l2.add(diff(v.l2_score, to_num(row[c_l2])), t_row, 1e-6);
    if (!skip_l1) {
      s_l1.add(diff(v.l1_score, to_num(row[c_l1])), t_row, 1e-6);
      s_s.add(diff(v.score, to_num(row[c_s])), t_row, 1e-6);
      const bool g_alert = to_bool(row[c_alert]);
      alerts += v.alert;
      golden_alerts += g_alert;
      if (v.alert != g_alert) {
        ++alert_bad;
        note(std::string("alert mismatch at t=") + row[c_t] + " (c++ " + (v.alert ? "1" : "0") + ", python " +
             (g_alert ? "1" : "0") + ")");
      }
    }
  }

  const bool ok = decode_bad == 0 && l0_bad == 0 && acc_bad == 0 && s_z.bad == 0 && s_l2.bad == 0 &&
                  (skip_l1 || (s_l1.bad == 0 && s_s.bad == 0 && alert_bad == 0 && s_r.bad == 0));
  std::printf("%s: %s rows=%zu cfg=%s input=%s\n", file.filename().string().c_str(), ok ? "PASS" : "FAIL",
              csv.rows.size(), cfg_path.filename().string().c_str(), use_packets ? "packet_hex" : "values");
  std::printf("  max|dz|=%.3g (t=%g, %lld over 1e-4)  max|dl2|=%.3g (t=%g)", s_z.max_err, s_z.at_t, s_z.bad,
              s_l2.max_err, s_l2.at_t);
  if (!skip_l1)
    std::printf("  max|dl1|=%.3g (t=%g)  max|dscore|=%.3g (t=%g)  l1r max rel err=%.3g%s", s_l1.max_err, s_l1.at_t,
                s_s.max_err, s_s.at_t, s_r.max_err, have_l1r ? "" : " (no l1r columns)");
  std::printf("\n  accepted mismatches=%lld  l0 mismatches=%lld  decode failures=%lld", acc_bad, l0_bad, decode_bad);
  if (!skip_l1)
    std::printf("  alert frames c++=%lld python=%lld mismatches=%lld", alerts, golden_alerts, alert_bad);
  std::printf("\n");
  for (const auto& n : notes) std::printf("    %s\n", n.c_str());
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string edge_hint;
  bool skip_l1 = false;
  std::vector<fs::path> files;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--edge-dir" && i + 1 < argc) edge_hint = argv[++i];
    else if (a == "--skip-l1") skip_l1 = true;
    else files.emplace_back(a);
  }

  unit_tests();
  std::printf("unit tests: %s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  bool ok = g_failures == 0;

  try {
    const fs::path edge_dir = find_edge_dir(edge_hint, argv[0]);
    if (files.empty()) {
      const fs::path td = edge_dir / "testdata";
      if (fs::exists(td))
        for (const auto& e : fs::directory_iterator(td)) {
          const std::string name = e.path().filename().string();
          if (name.rfind("golden_", 0) == 0 && e.path().extension() == ".csv") files.push_back(e.path());
        }
      std::sort(files.begin(), files.end());
      if (files.empty()) {
        std::printf("no golden files in %s\n", td.string().c_str());
        ok = false;
      }
    }
    for (const auto& f : files) {
      try {
        ok = check_golden(f, edge_dir, skip_l1) && ok;
      } catch (const std::exception& e) {
        std::printf("%s: FAIL (%s)\n", f.filename().string().c_str(), e.what());
        ok = false;
      }
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    ok = false;
  }
  std::printf("test_detector: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

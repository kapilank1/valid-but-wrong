// Per-frame latency of the C++ base detector on a replayed flight (default: the ses9_clean golden frames).
//
//   bench_detector [--edge-dir DIR] [--config CFG] [--golden CSV] [--passes N]
//
// Replays the frames N times (reset between passes) and reports p50 / p99 / max / mean microseconds per frame for
//   total = BaseDetector::update (L1 + L2 + scoring + alerting),
//   l1    = L1Checker::update alone, on the same accepted frames,
//   l2    = L2Model::zscore_last alone (input build + ONNX Runtime call + z), on frames with a full window,
// plus frames/s of the total replay loop. Ends with one JSON line.

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS  // std::getenv
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "config.hpp"
#include "detector.hpp"
#include "frame.hpp"
#include "l1.hpp"
#include "l2.hpp"

namespace fs = std::filesystem;
using namespace vbw;
using Clock = std::chrono::steady_clock;

namespace {

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

std::vector<uint8_t> from_hex(const std::string& h) {
  std::vector<uint8_t> out(h.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(std::stoul(h.substr(2 * i, 2), nullptr, 16));
  return out;
}

fs::path find_edge_dir(const std::string& hint, const char* argv0) {
  if (!hint.empty()) return fs::path(hint);
  if (const char* env = std::getenv("VBW_EDGE_DIR")) return fs::path(env);
  std::vector<fs::path> starts{fs::current_path()};
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

struct Item {
  Frame f;
  L0 l0;
};

// Frames of a golden CSV: from packet_hex (decode + LinkMonitor) when present, else t / seq / channel values.
std::vector<Item> load_frames(const fs::path& csv, const std::vector<std::string>& channels, uint8_t schema_id) {
  std::ifstream in(csv, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + csv.string());
  std::string line;
  std::getline(in, line);
  const auto header = split(line);
  std::map<std::string, size_t> col;
  for (size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
  const bool packets = col.count("packet_hex") > 0;
  std::vector<Item> items;
  LinkMonitor mon;
  while (std::getline(in, line)) {
    if (line.empty() || line == "\r") continue;
    const auto row = split(line);
    Item it;
    if (packets) {
      const auto pkt = from_hex(row.at(col.at("packet_hex")));
      if (!decode(pkt.data(), pkt.size(), it.f, it.l0)) continue;
    } else {
      it.f.schema_id = schema_id;
      it.f.t = std::strtod(row.at(col.at("t")).c_str(), nullptr);
      it.f.seq = static_cast<uint16_t>(std::strtol(row.at(col.at("seq")).c_str(), nullptr, 10));
      for (const auto& c : channels) it.f.values.push_back(std::strtod(row.at(col.at(c)).c_str(), nullptr));
    }
    it.l0 = mon.receive(it.f, it.l0);
    items.push_back(std::move(it));
  }
  return items;
}

struct Summary {
  size_t n = 0;
  double p50 = 0, p99 = 0, max = 0, mean = 0;
};

Summary summarize(std::vector<double> us) {
  Summary s;
  s.n = us.size();
  if (us.empty()) return s;
  std::sort(us.begin(), us.end());
  auto pct = [&us](double q) {
    const size_t k = static_cast<size_t>(std::ceil(q * static_cast<double>(us.size()))) - 1;
    return us[std::min(k, us.size() - 1)];
  };
  s.p50 = pct(0.50);
  s.p99 = pct(0.99);
  s.max = us.back();
  double sum = 0;
  for (double v : us) sum += v;
  s.mean = sum / static_cast<double>(us.size());
  return s;
}

double us_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

std::string json(const Summary& s) {
  char buf[200];
  std::snprintf(buf, sizeof buf, "{\"n\":%zu,\"p50\":%.2f,\"p99\":%.2f,\"max\":%.2f,\"mean\":%.2f}", s.n, s.p50, s.p99,
                s.max, s.mean);
  return buf;
}

void print_row(const char* name, const Summary& s) {
  std::printf("  %-6s n=%-8zu p50=%8.2f us  p99=%8.2f us  max=%9.2f us  mean=%8.2f us\n", name, s.n, s.p50, s.p99,
              s.max, s.mean);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string edge_hint, cfg_arg, golden_arg;
    int passes = 5;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--edge-dir" && i + 1 < argc) edge_hint = argv[++i];
      else if (a == "--config" && i + 1 < argc) cfg_arg = argv[++i];
      else if (a == "--golden" && i + 1 < argc) golden_arg = argv[++i];
      else if (a == "--passes" && i + 1 < argc) passes = std::max(1, std::atoi(argv[++i]));
      else {
        std::fprintf(stderr, "usage: bench_detector [--edge-dir DIR] [--config CFG] [--golden CSV] [--passes N]\n");
        return 2;
      }
    }
    fs::path cfg_path, golden;
    if (cfg_arg.empty() || golden_arg.empty()) {
      const fs::path edge = find_edge_dir(edge_hint, argv[0]);
      cfg_path = cfg_arg.empty() ? edge / "config" / "falcon9.cfg" : fs::path(cfg_arg);
      golden = golden_arg.empty() ? edge / "testdata" / "golden_ses9_clean.csv" : fs::path(golden_arg);
    } else {
      cfg_path = cfg_arg;
      golden = golden_arg;
    }

    const Config cfg = Config::load(cfg_path.string());
    const auto channels = cfg.words("channels");
    const auto items = load_frames(golden, channels, static_cast<uint8_t>(cfg.num_or("schema_id", 0)));
    if (items.empty()) throw std::runtime_error("no frames in " + golden.string());
    std::printf("bench_detector: %zu frames from %s, %d passes, config %s\n", items.size(),
                golden.filename().string().c_str(), passes, cfg_path.filename().string().c_str());

    // ---- total: BaseDetector::update. One untimed pass first (page-in, ORT first-run allocations).
    BaseDetector det(cfg);
    std::vector<char> accepted(items.size(), 0);
    for (size_t i = 0; i < items.size(); ++i) accepted[i] = det.update(items[i].f, items[i].l0).accepted;
    std::vector<double> total;
    total.reserve(items.size() * static_cast<size_t>(passes));
    double checksum = 0, loop_us = 0;
    long long alert_frames = 0;
    for (int p = 0; p < passes; ++p) {
      det.reset();
      const auto start = Clock::now();
      for (const auto& it : items) {
        const auto a = Clock::now();
        const Verdict v = det.update(it.f, it.l0);
        const auto b = Clock::now();
        total.push_back(us_between(a, b));
        checksum += v.score;
        alert_frames += v.alert;
      }
      loop_us += us_between(start, Clock::now());
    }

    // ---- L1 alone, on the frames the detector accepts.
    L1Checker l1(cfg);
    std::vector<double> l1_us;
    l1_us.reserve(items.size() * static_cast<size_t>(passes));
    for (int p = 0; p < passes; ++p) {
      l1.reset();
      for (size_t i = 0; i < items.size(); ++i) {
        if (!accepted[i]) continue;
        const auto a = Clock::now();
        const L1Result r = l1.update(items[i].f.values, items[i].f.t);
        const auto b = Clock::now();
        l1_us.push_back(us_between(a, b));
        checksum += r.max_ratio;
      }
    }

    // ---- L2 alone: zscore_last on every full window of accepted frames.
    std::vector<double> l2_us;
    if (det.has_l2()) {
      L2Model l2(cfg);
      const size_t C = channels.size(), need = static_cast<size_t>(l2.needs());
      std::vector<double> win, z(C);
      win.reserve(need * C);
      for (int p = 0; p < passes; ++p) {
        std::deque<const std::vector<double>*> hist;
        for (size_t i = 0; i < items.size(); ++i) {
          if (!accepted[i]) continue;
          hist.push_back(&items[i].f.values);
          if (hist.size() > need) hist.pop_front();
          if (hist.size() < need) continue;
          win.clear();
          for (const auto* x : hist) win.insert(win.end(), x->begin(), x->begin() + static_cast<std::ptrdiff_t>(C));
          const auto a = Clock::now();
          l2.zscore_last(win.data(), z.data());
          const auto b = Clock::now();
          l2_us.push_back(us_between(a, b));
          checksum += z[0];
        }
      }
    }

    const Summary st = summarize(total), s1 = summarize(l1_us), s2 = summarize(l2_us);
    const double fps = static_cast<double>(total.size()) / (loop_us * 1e-6);
    print_row("total", st);
    print_row("l1", s1);
    print_row("l2", s2);
    std::printf("  frames/s=%.0f  alert frames per pass=%lld  (checksum %.6g)\n", fps, alert_frames / passes, checksum);
    std::printf("{\"type\":\"bench_detector\",\"frames\":%zu,\"passes\":%d,\"total_us\":%s,\"l1_us\":%s,\"l2_us\":%s,"
                "\"frames_per_s\":%.1f,\"alert_frames_per_pass\":%lld,\"l2\":%s}\n",
                items.size(), passes, json(st).c_str(), json(s1).c_str(), json(s2).c_str(), fps,
                alert_frames / passes, det.has_l2() ? "true" : "false");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_detector: %s\n", e.what());
    return 1;
  }
}

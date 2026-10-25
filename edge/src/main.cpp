// vbw_edge: native ground-station monitor. Listens for vbw telemetry frames on UDP, runs L0 link checks and the
// base detector (L1 physics + L2 GRU + k-of-n alerting) per vehicle, and reports alerts and per-frame latency.
//
//   vbw_edge --config edge/config/falcon9.cfg [--port 14650] [--bind 127.0.0.1] [--jsonl out.jsonl]
//            [--log-frames] [--idle-exit SECONDS] [--max-frames N] [--quiet]
//
// See edge/SPEC.md section 5 and edge/README.md.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS  // fopen / std::getenv
#endif

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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
#include "detector.hpp"
#include "frame.hpp"
#include "udp.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::atomic<bool> g_stop{false};
std::atomic<bool> g_done{false};

#ifdef _WIN32
BOOL WINAPI on_console_ctrl(DWORD type) {
  g_stop = true;
  if (type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) {
    // Windows terminates the process when this handler returns: give the main loop time to write the summary.
    for (int i = 0; i < 60 && !g_done; ++i) Sleep(50);
  }
  return TRUE;
}
#else
extern "C" void on_signal(int) { g_stop = true; }
#endif

void install_stop_handlers() {
#ifdef _WIN32
  SetConsoleCtrlHandler(on_console_ctrl, TRUE);
#else
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
#endif
}

// ------------------------------------------------------------------------------------------------ options

struct Options {
  std::string config;
  std::string bind = "127.0.0.1";
  int port = 14650;
  std::string jsonl;
  bool log_frames = false;
  double idle_exit_s = 0;     // 0 = never
  uint64_t max_frames = 0;    // 0 = unlimited (counts datagrams)
  bool quiet = false;
};

const char* kUsage =
    "usage: vbw_edge --config edge/config/falcon9.cfg [--port 14650] [--bind 127.0.0.1] [--jsonl out.jsonl]\n"
    "                [--log-frames] [--idle-exit SECONDS] [--max-frames N] [--quiet]\n"
    "\n"
    "  --config PATH       edge config written by scripts/export_edge_config.py (required)\n"
    "  --port N            UDP port to listen on (default 14650; 0 = any free port, printed at startup)\n"
    "  --bind ADDR         local address to bind (default 127.0.0.1; 0.0.0.0 = all interfaces)\n"
    "  --jsonl PATH        write alert rising edges (and with --log-frames every frame) plus the summary as JSONL\n"
    "  --log-frames        also write one JSONL record per processed frame\n"
    "  --idle-exit S       exit when no datagram arrived for S seconds (the clock starts at the first datagram)\n"
    "  --max-frames N      exit after N datagrams\n"
    "  --quiet             no ALERT lines or warnings on the console (the summary is still printed)\n";

[[noreturn]] void usage_error(const std::string& msg) {
  std::fprintf(stderr, "vbw_edge: %s\n\n%s", msg.c_str(), kUsage);
  std::exit(2);
}

double parse_double_arg(const std::string& opt, const std::string& v) {
  double out = 0;
  auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
  if (ec != std::errc() || p != v.data() + v.size() || !std::isfinite(out) || out < 0) {
    usage_error(opt + " expects a non-negative number, got '" + v + "'");
  }
  return out;
}

uint64_t parse_uint_arg(const std::string& opt, const std::string& v, uint64_t max) {
  uint64_t out = 0;
  auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
  if (ec != std::errc() || p != v.data() + v.size() || out > max) {
    usage_error(opt + " expects an integer in [0, " + std::to_string(max) + "], got '" + v + "'");
  }
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    std::string val;
    bool has_val = false;
    if (a.rfind("--", 0) == 0) {
      const size_t eq = a.find('=');
      if (eq != std::string::npos) {
        val = a.substr(eq + 1);
        a = a.substr(0, eq);
        has_val = true;
      }
    }
    auto value = [&]() -> std::string {
      if (has_val) return val;
      if (i + 1 >= argc) usage_error(a + " needs a value");
      return argv[++i];
    };
    auto no_value = [&]() {
      if (has_val) usage_error(a + " takes no value");
    };
    if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      std::exit(0);
    } else if (a == "--config") {
      o.config = value();
    } else if (a == "--port") {
      o.port = static_cast<int>(parse_uint_arg(a, value(), 65535));
    } else if (a == "--bind") {
      o.bind = value();
    } else if (a == "--jsonl") {
      o.jsonl = value();
    } else if (a == "--log-frames") {
      no_value();
      o.log_frames = true;
    } else if (a == "--idle-exit") {
      o.idle_exit_s = parse_double_arg(a, value());
    } else if (a == "--max-frames") {
      o.max_frames = parse_uint_arg(a, value(), UINT64_MAX);
    } else if (a == "--quiet") {
      no_value();
      o.quiet = true;
    } else {
      usage_error("unknown argument '" + a + "'");
    }
  }
  if (o.config.empty()) usage_error("--config is required");
  return o;
}

// ------------------------------------------------------------------------------------------------ JSON

void json_num(std::string& s, double v) {
  if (!std::isfinite(v)) {   // JSON has no NaN / inf
    s += "null";
    return;
  }
  char buf[32];
  auto r = std::to_chars(buf, buf + sizeof buf, v);   // shortest round-trip form, like Python repr()
  s.append(buf, r.ptr);
}

void json_int(std::string& s, uint64_t v) {
  char buf[24];
  auto r = std::to_chars(buf, buf + sizeof buf, v);
  s.append(buf, r.ptr);
}

void json_str(std::string& s, std::string_view v) {
  static const char* hex = "0123456789abcdef";
  s += '"';
  for (const char ch : v) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': s += "\\\""; break;
      case '\\': s += "\\\\"; break;
      case '\b': s += "\\b"; break;
      case '\f': s += "\\f"; break;
      case '\n': s += "\\n"; break;
      case '\r': s += "\\r"; break;
      case '\t': s += "\\t"; break;
      default:
        if (c < 0x20 || c == 0x7F) {
          s += "\\u00";
          s += hex[c >> 4];
          s += hex[c & 0xF];
        } else {
          s += ch;   // UTF-8 passes through unchanged
        }
    }
  }
  s += '"';
}

void json_vid(std::string& s, uint16_t vid) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "\"0x%04x\"", vid);
  s += buf;
}

// ------------------------------------------------------------------------------------------------ latency

// Per-frame latency in ns: exact count / mean / max, percentiles from a uniform reservoir (exact up to kCap frames).
class LatencyStats {
 public:
  static constexpr size_t kCap = size_t{1} << 21;   // 2M samples (8 MB), reserved up front

  LatencyStats() { sample_.reserve(kCap); }

  void add(uint64_t ns) {
    ++count_;
    sum_ns_ += ns;
    if (ns > max_ns_) max_ns_ = ns;
    const uint32_t v = ns > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(ns);
    if (sample_.size() < kCap) {
      sample_.push_back(v);
    } else {
      const uint64_t j = rng_() % count_;   // Algorithm R
      if (j < kCap) sample_[static_cast<size_t>(j)] = v;
    }
  }

  uint64_t count() const { return count_; }
  uint64_t sum_ns() const { return sum_ns_; }
  uint64_t max_ns() const { return max_ns_; }
  size_t samples() const { return sample_.size(); }

  // Percentiles (numpy's default linear interpolation) in ns. Sorts the sample; call at the end.
  double percentile(double p) {
    if (sample_.empty()) return 0;
    if (!sorted_) {
      std::sort(sample_.begin(), sample_.end());
      sorted_ = true;
    }
    const double idx = p / 100.0 * static_cast<double>(sample_.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(lo + 1, sample_.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return static_cast<double>(sample_[lo]) + (static_cast<double>(sample_[hi]) - static_cast<double>(sample_[lo])) * frac;
  }

 private:
  std::vector<uint32_t> sample_;
  uint64_t count_ = 0, sum_ns_ = 0, max_ns_ = 0;
  bool sorted_ = false;
  std::mt19937_64 rng_{0x5EEDu};
};

// ------------------------------------------------------------------------------------------------ monitor

struct Vehicle {
  uint8_t schema_id = 0;
  uint16_t vid = 0;
  vbw::LinkMonitor link;
  std::unique_ptr<vbw::BaseDetector> det;
  bool alerting = false;
  uint64_t frames = 0, alerts = 0, l0_failures = 0;
};

struct Counters {
  uint64_t packets = 0;           // datagrams received
  uint64_t decode_errors = 0;     // structurally unparseable (too short, unknown schema / channel count, length)
  uint64_t truncated = 0;         // datagrams larger than the receive buffer
  uint64_t other_schema = 0;      // well-formed frames of a schema other than the config's
  uint64_t frames = 0;            // frames run through a vehicle's LinkMonitor + detector
  uint64_t accepted = 0;          // frames the detector accepted into its physics history
  uint64_t l0_failures = 0;       // frames (of our schema) failing any L0 check, incl. unattributed ones
  uint64_t unattributed = 0;      // sync/CRC failures from an unknown vehicle key: no state is created for them
  uint64_t alerts = 0;            // alert rising edges
  double setup_ms = 0;            // time spent creating per-vehicle detectors (excluded from frame latency)
};

class JsonlWriter {
 public:
  explicit JsonlWriter(const std::string& path) {
    if (path.empty()) return;
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot open --jsonl file '" + path + "' for writing");
    std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
  }
  ~JsonlWriter() {
    if (f_) std::fclose(f_);
  }
  JsonlWriter(const JsonlWriter&) = delete;
  JsonlWriter& operator=(const JsonlWriter&) = delete;
  bool enabled() const { return f_ != nullptr; }
  void write_line(const std::string& line) {
    if (!f_) return;
    std::fwrite(line.data(), 1, line.size(), f_);
    std::fputc('\n', f_);
  }
  void flush() {
    if (f_) std::fflush(f_);
  }

 private:
  FILE* f_ = nullptr;
};

std::string format_summary(Counters& c, size_t vehicles, LatencyStats& lat, double span_s, double elapsed_s,
                           const char* exit_reason) {
  std::string s;
  s.reserve(512);
  s += "{\"type\":\"summary\",\"frames\":";
  json_int(s, c.frames);
  s += ",\"vehicles\":";
  json_int(s, vehicles);
  s += ",\"alerts\":";
  json_int(s, c.alerts);
  s += ",\"l0_failures\":";
  json_int(s, c.l0_failures);
  s += ",\"latency_us\":{\"p50\":";
  json_num(s, lat.percentile(50) / 1000.0);
  s += ",\"p99\":";
  json_num(s, lat.percentile(99) / 1000.0);
  s += ",\"max\":";
  json_num(s, static_cast<double>(lat.max_ns()) / 1000.0);
  s += ",\"mean\":";
  json_num(s, lat.count() ? static_cast<double>(lat.sum_ns()) / static_cast<double>(lat.count()) / 1000.0 : 0.0);
  s += ",\"samples\":";
  json_int(s, lat.samples());
  s += "},\"frames_per_s\":";
  json_num(s, span_s > 0 ? static_cast<double>(c.frames) / span_s : 0.0);
  s += ",\"busy_frames_per_s\":";
  json_num(s, lat.sum_ns() ? static_cast<double>(lat.count()) * 1e9 / static_cast<double>(lat.sum_ns()) : 0.0);
  s += ",\"packets\":";
  json_int(s, c.packets);
  s += ",\"accepted\":";
  json_int(s, c.accepted);
  s += ",\"decode_errors\":";
  json_int(s, c.decode_errors);
  s += ",\"truncated\":";
  json_int(s, c.truncated);
  s += ",\"other_schema\":";
  json_int(s, c.other_schema);
  s += ",\"unattributed_l0\":";
  json_int(s, c.unattributed);
  s += ",\"vehicle_setup_ms\":";
  json_num(s, c.setup_ms);
  s += ",\"stream_s\":";
  json_num(s, span_s);
  s += ",\"elapsed_s\":";
  json_num(s, elapsed_s);
  s += ",\"exit\":";
  json_str(s, exit_reason);
  s += "}";
  return s;
}

int run(const Options& opt) {
  const vbw::Config cfg = vbw::Config::load(opt.config);
  const std::string schema = cfg.words("schema").at(0);
  const double sid = cfg.num("schema_id");
  if (!(sid >= 0 && sid <= 255 && sid == std::floor(sid))) {
    throw std::runtime_error("config " + opt.config + ": schema_id must be an integer in [0, 255]");
  }
  const auto schema_id = static_cast<uint8_t>(sid);
  const size_t n_channels = cfg.words("channels").size();
  const int wire_channels = vbw::expected_channels(schema_id);
  if (wire_channels < 0) {
    throw std::runtime_error("config " + opt.config + ": schema_id " + std::to_string(schema_id) +
                             " is not a known wire schema (falcon9 = 1, quad = 2, spacecraft = 3)");
  }
  if (static_cast<size_t>(wire_channels) != n_channels) {
    throw std::runtime_error("config " + opt.config + ": " + std::to_string(n_channels) + " channels, but schema " +
                             std::to_string(schema_id) + " frames carry " + std::to_string(wire_channels));
  }

  // Build one detector up front: validates the config and loads the ONNX model before we listen. The first
  // vehicle gets it.
  std::unique_ptr<vbw::BaseDetector> spare = std::make_unique<vbw::BaseDetector>(cfg);
  const size_t n_rules = spare->rule_names().size();
  const bool has_l2 = spare->has_l2();

  JsonlWriter jsonl(opt.jsonl);
  vbw::UdpReceiver rx(opt.bind, static_cast<uint16_t>(opt.port));
  const int tick_ms = 100;   // Ctrl+C / idle-exit check interval while no datagrams arrive
  rx.set_timeout_ms(tick_ms);

#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
#endif
  std::fprintf(stderr,
               "vbw_edge: %s (schema %u, %zu channels, %zu L1 rules, L2 %s) listening on udp://%s:%u "
               "(SO_RCVBUF %d bytes)\n",
               schema.c_str(), schema_id, n_channels, n_rules, has_l2 ? "on" : "off", rx.host().c_str(), rx.port(),
               rx.rcvbuf_bytes());
  std::fflush(stderr);

  // Wall-clock receive timestamps from the steady clock (one system_clock read at startup).
  const auto sys0 = std::chrono::system_clock::now();
  const auto steady0 = Clock::now();
  const int64_t sys0_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(sys0.time_since_epoch()).count();

  std::unordered_map<uint32_t, std::unique_ptr<Vehicle>> vehicles;
  vehicles.reserve(1024);
  Counters c;
  LatencyStats lat;
  std::vector<uint8_t> buf(65536);
  vbw::Frame frame;
  frame.values.reserve(256);
  vbw::L0 l0;
  std::string reason, line;
  line.reserve(1024);
  uint64_t decode_warnings = 0;

  bool any_packet = false;
  Clock::time_point first_frame{}, last_frame{}, last_packet = steady0;
  const char* exit_reason = "signal";

  while (true) {
    if (g_stop) {
      exit_reason = "signal";
      break;
    }
    size_t len = 0;
    const auto st = rx.receive(buf.data(), buf.size(), len);
    if (st == vbw::UdpReceiver::Status::Timeout) {
      jsonl.flush();
      if (opt.idle_exit_s > 0 && any_packet &&
          std::chrono::duration<double>(Clock::now() - last_packet).count() >= opt.idle_exit_s) {
        exit_reason = "idle";
        break;
      }
      continue;
    }
    if (st == vbw::UdpReceiver::Status::Error) {
      std::fprintf(stderr, "vbw_edge: receive failed: %s\n", rx.last_error().c_str());
      exit_reason = "socket-error";
      break;
    }

    // ---- hot path: latency is measured from here (recvfrom returned) to the verdict being ready.
    const auto t0 = Clock::now();
    last_packet = t0;
    any_packet = true;
    ++c.packets;
    if (rx.truncated()) ++c.truncated;
    const bool limit = opt.max_frames && c.packets >= opt.max_frames;

    if (!vbw::decode(buf.data(), len, frame, l0, &reason)) {
      ++c.decode_errors;
      if (!opt.quiet && decode_warnings < 5) {
        std::fprintf(stderr, "vbw_edge: dropped undecodable datagram (%s)%s\n", reason.c_str(),
                     decode_warnings == 4 ? "; further decode errors are only counted" : "");
      }
      ++decode_warnings;
      if (limit) {
        exit_reason = "max-frames";
        break;
      }
      continue;
    }
    if (frame.schema_id != schema_id) {
      ++c.other_schema;
      if (limit) {
        exit_reason = "max-frames";
        break;
      }
      continue;
    }

    const uint32_t key = (static_cast<uint32_t>(frame.schema_id) << 16) | frame.vid_hash;
    auto it = vehicles.find(key);
    Clock::duration setup{0};
    if (it == vehicles.end()) {
      if (!(l0.sync_ok && l0.crc_ok)) {
        // A corrupted header may carry a garbage vid_hash: don't create a phantom vehicle for it.
        ++c.unattributed;
        ++c.l0_failures;
        if (limit) {
          exit_reason = "max-frames";
          break;
        }
        continue;
      }
      const auto s0 = Clock::now();
      auto v = std::make_unique<Vehicle>();
      v->schema_id = frame.schema_id;
      v->vid = frame.vid_hash;
      v->det = spare ? std::move(spare) : std::make_unique<vbw::BaseDetector>(cfg);
      it = vehicles.emplace(key, std::move(v)).first;
      setup = Clock::now() - s0;
      c.setup_ms += std::chrono::duration<double, std::milli>(setup).count();
    }
    Vehicle& veh = *it->second;
    l0 = veh.link.receive(frame, l0);
    const vbw::Verdict verdict = veh.det->update(frame, l0);
    const auto t1 = Clock::now();
    // ---- verdict ready

    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>((t1 - t0) - setup).count();
    const uint64_t lat_ns = ns > 0 ? static_cast<uint64_t>(ns) : 0;
    lat.add(lat_ns);
    if (c.frames == 0) first_frame = t0;
    last_frame = t0;
    ++c.frames;
    ++veh.frames;
    if (verdict.accepted) ++c.accepted;
    if (!l0.ok()) {
      ++c.l0_failures;
      ++veh.l0_failures;
    }
    const bool rising = verdict.alert && !veh.alerting;
    veh.alerting = verdict.alert;

    if (rising) {
      ++c.alerts;
      ++veh.alerts;
      const double lat_us = static_cast<double>(lat_ns) / 1000.0;
      if (!opt.quiet) {
        std::printf("ALERT vid=0x%04X t=%.2f score=%.2f latency_us=%lld %s\n", veh.vid, verdict.t, verdict.score,
                    static_cast<long long>(std::llround(lat_us)), verdict.explanation.c_str());
        std::fflush(stdout);
      }
      if (jsonl.enabled()) {
        const int64_t recv_ns =
            sys0_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - steady0).count();
        line.clear();
        line += "{\"type\":\"alert\",\"vid\":";
        json_vid(line, veh.vid);
        line += ",\"t\":";
        json_num(line, verdict.t);
        line += ",\"seq\":";
        json_int(line, verdict.seq);
        line += ",\"score\":";
        json_num(line, verdict.score);
        line += ",\"l1_score\":";
        json_num(line, verdict.l1_score);
        line += ",\"l2_score\":";
        json_num(line, verdict.l2_score);
        line += ",\"explanation\":";
        json_str(line, verdict.explanation);
        line += ",\"recv_ns\":";
        json_int(line, static_cast<uint64_t>(recv_ns));
        line += ",\"latency_us\":";
        json_num(line, lat_us);
        line += "}";
        jsonl.write_line(line);
        jsonl.flush();
      }
    }
    if (opt.log_frames && jsonl.enabled()) {
      line.clear();
      line += "{\"type\":\"frame\",\"vid\":";
      json_vid(line, veh.vid);
      line += ",\"t\":";
      json_num(line, verdict.t);
      line += ",\"seq\":";
      json_int(line, verdict.seq);
      line += ",\"l0_ok\":";
      line += l0.ok() ? "true" : "false";
      line += ",\"accepted\":";
      line += verdict.accepted ? "true" : "false";
      line += ",\"score\":";
      json_num(line, verdict.score);
      line += ",\"l1_score\":";
      json_num(line, verdict.l1_score);
      line += ",\"l2_score\":";
      json_num(line, verdict.l2_score);
      line += ",\"alert\":";
      line += verdict.alert ? "true" : "false";
      line += ",\"latency_us\":";
      json_num(line, static_cast<double>(lat_ns) / 1000.0);
      line += "}";
      jsonl.write_line(line);
    }
    if (limit) {
      exit_reason = "max-frames";
      break;
    }
  }

  const double span_s = c.frames >= 2 ? std::chrono::duration<double>(last_frame - first_frame).count() : 0.0;
  const double elapsed_s = std::chrono::duration<double>(Clock::now() - steady0).count();
  const std::string summary = format_summary(c, vehicles.size(), lat, span_s, elapsed_s, exit_reason);
  jsonl.write_line(summary);
  jsonl.flush();
  std::printf("%s\n", summary.c_str());
  std::fflush(stdout);
  if (!opt.quiet) {
    std::fprintf(stderr,
                 "vbw_edge: %llu frames from %zu vehicle(s), %llu alert(s), %llu L0 failure(s), %llu undecodable; "
                 "latency p50 %.1f us, p99 %.1f us, max %.1f us; %.0f frames/s (exit: %s)\n",
                 static_cast<unsigned long long>(c.frames), vehicles.size(),
                 static_cast<unsigned long long>(c.alerts), static_cast<unsigned long long>(c.l0_failures),
                 static_cast<unsigned long long>(c.decode_errors), lat.percentile(50) / 1000.0,
                 lat.percentile(99) / 1000.0, static_cast<double>(lat.max_ns()) / 1000.0,
                 span_s > 0 ? static_cast<double>(c.frames) / span_s : 0.0, exit_reason);
  }
  return std::string(exit_reason) == "socket-error" ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
  install_stop_handlers();
  int rc = 1;
  try {
    rc = run(opt);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vbw_edge: error: %s\n", e.what());
    rc = 1;
  }
  g_done = true;
  return rc;
}

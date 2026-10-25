# vbw_edge: C++ edge monitor — build contract

> **Status (2026-10-24 15:35, PAUSED by the user):** the four build agents stopped on an API session limit; their work
> is on disk. `powershell -ExecutionPolicy Bypass -File edge\build.ps1 -Test` builds everything and **3/3 CTest suites
> pass** (test_frame, test_config, test_detector). test_detector proves **bit-exact parity** with the Python
> BaseDetector on all 4 golden sets (16,766 frames, max error 0, alert frames identical); `bench_detector`: 30 µs per
> frame p50 (L1 3.1 µs, L2 26.4 µs), 32,550 frames/s (`results/edge_bench.json`).
> **Not done:** `edge/tests/test_l1.cpp` (per-rule parity; L1 is already covered end to end by test_detector),
> `tests/test_edge_*.py` (Python tests for the agent-A scripts), `edge/README.md`, a live end-to-end UDP run
> (`uv run python scripts/edge_eval.py` with `edge/build/vbw_edge.exe` + `scripts/udp_replay.py` → `results/edge_eval.json`),
> and phase 2 (MAVLink input, L3 in C++, Linux/Docker). Agent A's scripts exist (`export_edge_config.py`,
> `edge_golden.py`, `udp_replay.py`, `edge_eval.py`) but are untested end to end against the binary.

`vbw_edge` is a native C++ ground-station monitor. It listens on UDP for the same CCSDS-style telemetry frames the
Python pipeline encodes (`vbw/link/codec.py`), runs L0 link checks and the **base detector** (L1 physics rules + L2
GRU via ONNX Runtime + k-of-n alerting) per vehicle, and reports alerts plus per-frame latency. It must reproduce the
Python `BaseDetector` (`vbw/detector/base.py`) **exactly** (same ratios, scores and alerts on the same frames), which is
proven by golden-file parity tests. Many vehicles can stream to one process at once.

The Python code is the reference implementation for every number. When this spec and the Python disagree, the Python
wins; note the discrepancy in your report.

## Toolchain (already set up by the orchestrator)

- MSVC 14.51 (VS Build Tools 2026), CMake + Ninja bundled with it, Windows 11 SDK 10.0.26100 (being installed).
  Load the compiler environment with `vcvars64.bat` (`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
- ONNX Runtime 1.31.0 CPU, prebuilt: `edge/third_party/onnxruntime-win-x64-1.31.0/{include,lib}` (`onnxruntime.lib`,
  `onnxruntime.dll`; copy the DLL next to every executable).
- MAVLink v2 C headers: `edge/third_party/mavlink` (header-only; phase 2 only).
- C++20. No other third-party dependencies (no JSON library, no test framework: tests are small executables that
  return non-zero on failure, registered with CTest).
- `edge/third_party/` and `edge/build/` are gitignored.

## Layout and ownership (one owner per file; don't edit files you don't own — report needed changes instead)

| Path | Owner |
|---|---|
| `edge/SPEC.md` | orchestrator |
| `edge/CMakeLists.txt`, `edge/build.ps1`, `edge/src/main.cpp`, `edge/src/frame.{hpp,cpp}`, `edge/src/udp.{hpp,cpp}`, `edge/src/config.{hpp,cpp}`, `edge/tests/test_frame.cpp`, `edge/tests/test_config.cpp` | **agent B** (C++ platform) |
| `edge/src/l1.{hpp,cpp}`, `edge/tests/test_l1.cpp` | **agent C** (C++ L1 physics) |
| `edge/src/l2.{hpp,cpp}`, `edge/src/alerting.{hpp,cpp}`, `edge/src/detector.{hpp,cpp}`, `edge/tests/test_detector.cpp` | **agent D** (C++ L2 + alerting + detector) |
| `scripts/export_edge_config.py`, `scripts/edge_golden.py`, `scripts/udp_replay.py`, `scripts/edge_eval.py`, `edge/config/*.cfg`, `edge/testdata/*`, `tests/test_edge_*.py` | **agent A** (Python side) |

## 1. Wire format (UDP payload = exactly one frame, big-endian) — from `vbw/link/codec.py`

```
sync u32 = 0x1ACFFC1D | version u8 = 1 | schema_id u8 | vid_hash u16 | seq u16 | t f64 | n u8 | n × f32 values | crc u16
```
`crc` = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xorout) over every byte before it.
`vid_hash` = CRC-16/CCITT-FALSE of the vehicle_id UTF-8 string. Vehicles are keyed by `(schema_id, vid_hash)`.
Check value for CRC-16/CCITT-FALSE("123456789") = 0x29B1.
L0 per vehicle (`LinkMonitor`): sync_ok, crc_ok, seq_ok (seq == last+1 mod 65536), time_ok (t > last t).
The detector skips frames with a sync/CRC failure or non-finite values, and frames whose t <= last accepted t
(exactly as `BaseDetector._score`).

Values are float32 on the wire; the detector works in double on those float32 values (`wire_round` in Python).

## 2. Config file (`edge/config/<schema>.cfg`), written by `scripts/export_edge_config.py`

Plain text, one entry per line: `key value value ...`; `#` starts a comment; blank lines ignored. A line whose values
all parse as numbers (`strtod`, including `nan`, `inf`, `-inf`) is numeric; otherwise its tokens are strings.
Paths are relative to the config file's directory.

| Key | Values |
|---|---|
| `schema`, `schema_id`, `rate_hz` | `falcon9`, `1`, `10` |
| `channels` | channel names in wire order |
| `l1.rules` | rule names in `RULES_BY_SCHEMA[schema]` order |
| `l1.threshold.<rule>` | calibrated threshold (`models/l1_thresholds.json`, defaults merged as `L1Config.load` does) |
| `l1.lookback.<rule>` | `Rule.lookback` (frames of history before the newest frame) |
| `l1.params.<dotted.path>` | every leaf of the L1 `params` dict, flattened: nested dicts → dotted keys, lists → value lists, scalars → one value (e.g. `l1.params.atmosphere.alt`, `l1.params.corridor.altitude_lo`, `l1.params.corridor.t_all_end`) |
| `l1.const.<NAME>` | module constants the rules use (`F9_WINDOW`, `F9_LONG_WINDOW`, `F9_SMOOTH`, `F9_MIN_SPEED`, `F9_MIN_FLIGHT_SPEED`, `F9_MIN_Q_ALTITUDE`, `G0`, `R_EARTH`, ...), so C++ has no magic numbers |
| `l2.onnx` | path to the L2 ONNX model |
| `l2.window` | `L2_WINDOW` (20); the model needs `window + 2` frames |
| `l2.mean`, `l2.std`, `l2.dstd`, `l2.res_median`, `l2.res_mad` | per-channel arrays from `models/l2_<schema>_norm.json` |
| `l2.inst_threshold`, `l2.ewma_alpha`, `l2.ewma_threshold`, `l2.z_clip` | scalars |
| `alert.tau`, `alert.k`, `alert.n`, `alert.clear_frames`, `alert.clear_ratio`, `alert.warmup` | `AlertConfig.load(schema)` |

## 3. Golden parity file (`edge/testdata/golden_<name>.csv`), written by `scripts/edge_golden.py`

Generated by running the **Python** `BaseDetector` frame by frame on frames that went through `codec.encode` →
`codec.decode` (so values are float32-rounded exactly as the C++ sees them). One header row, then one row per frame.
Numbers printed with `repr()` (round-trip exact); NaN as `nan`.

Columns: `t, seq, <channel values in wire order>, l0_ok, accepted, l1r.<rule>... , l1_score, l2z.<channel>... ,
l2_score, score, alert`
- `l1r.<rule>` = signed ratio residual/threshold for the newest frame, `0` when the rule isn't evaluable yet
  (exactly `L1Result.ratios`).
- `l2z.<channel>` = `BaseDetector.last_z` (0 while history is short).
- `accepted` = whether the detector accepted the frame into its history.

Required golden sets: `ses9_clean` (falcon9:ses-9, no attack), `ses9_bias` (bias intensity 1 at T+190 s),
`ses9_ppo` (PPO r1 at T+190 s), plus `link_faults` (a short clean stream with injected bad CRC, bad sync, seq jump,
non-monotonic t and a NaN value).

Parity tolerances: L1 ratios abs 1e-9 (relative 1e-9 for large values), L2 z abs 1e-4, scores abs 1e-6, alerts exact.

## 4. C++ interfaces (namespace `vbw`)

```cpp
// config.hpp  (agent B)
struct Config {
  std::map<std::string, std::vector<double>> nums;
  std::map<std::string, std::vector<std::string>> strs;
  std::string base_dir;                                      // directory of the config file
  static Config load(const std::string& path);               // throws std::runtime_error with line number
  bool has(const std::string& key) const;
  const std::vector<double>& vec(const std::string& key) const;          // throws if missing / not numeric
  double num(const std::string& key) const;                              // exactly one value
  double num_or(const std::string& key, double fallback) const;
  const std::vector<std::string>& words(const std::string& key) const;   // throws if missing / numeric
  std::string path(const std::string& key) const;            // words(key)[0] resolved against base_dir
  std::vector<std::string> keys_with_prefix(const std::string& prefix) const;
};

// frame.hpp  (agent B)
uint16_t crc16_ccitt(const uint8_t* data, size_t n, uint16_t crc = 0xFFFF);
uint16_t vehicle_hash(const std::string& vehicle_id);
struct L0 { bool sync_ok = true, crc_ok = true, seq_ok = true, time_ok = true;
            bool ok() const { return sync_ok && crc_ok && seq_ok && time_ok; } };
struct Frame { uint8_t version, schema_id; uint16_t vid_hash, seq; double t; std::vector<double> values; };
// Parses one packet; returns false (with reason) only if structurally unparseable. Integrity failures go in l0.
bool decode(const uint8_t* data, size_t n, Frame& out, L0& l0, std::string* reason = nullptr);
std::vector<uint8_t> encode(const Frame& f);                 // for tests and the replay tool parity
class LinkMonitor { public: void reset(); L0 receive(const Frame& f, L0 l0); };   // seq / time continuity

// l1.hpp  (agent C)
class FrameBuffer {                       // ring buffer, contiguous last-n view (like physics.FrameBuffer)
 public:
  FrameBuffer(size_t channels, size_t capacity);
  void clear(); void append(const std::vector<double>& x, double t); size_t size() const; size_t channels() const;
  // Copies the last n (<= size) frames: X row-major [n][channels], t [n].
  void last(size_t n, std::vector<double>& X, std::vector<double>& t) const;
};
struct L1Result { std::vector<double> ratios; double max_ratio = 0;              // ratios in rule order
                  std::vector<int> violated; std::string message; };            // message of the worst violation
class L1Checker {
 public:
  explicit L1Checker(const Config& cfg);
  void reset();
  L1Result update(const std::vector<double>& x, double t);   // append + evaluate every rule on the newest frame
  const FrameBuffer& buffer() const;
  const std::vector<std::string>& rule_names() const;
};

// alerting.hpp / l2.hpp / detector.hpp  (agent D)
double soft_ratio(double r);                                  // r^2 / (1 + r^2)
double combine(std::initializer_list<double> scores);         // noisy-OR, each clipped to [0, 1]
struct AlertConfig { double tau; int k, n, clear_frames; double clear_ratio; int warmup;
                     static AlertConfig from(const Config&); };
class AlertState { public: explicit AlertState(AlertConfig); void reset(); bool update(double score); };
class L2Model { public: explicit L2Model(const Config&); int needs() const;
                // z-scores of the newest frame of X [needs][C] row-major
                std::vector<double> zscore_last(const std::vector<double>& X) const; };
struct Verdict { double t = 0; uint16_t seq = 0; bool accepted = false; L0 l0;
                 std::vector<double> l1_ratios, l2_z; double l1_score = 0, l2_score = 0, score = 0;
                 bool alert = false; std::string explanation; };
class BaseDetector {
 public:
  explicit BaseDetector(const Config& cfg);
  void reset();
  Verdict update(const Frame& f, const L0& l0);              // mirrors BaseDetector.update in vbw/detector/base.py
};
```
ONNX Runtime: one process-wide `Ort::Env`; sessions with `intra_op_num_threads = 1`, `inter_op_num_threads = 1`
(as in Python). The L2 input name is `x` ([1, window, 2·C] float32) and the output `delta` ([1, C]).

## 5. `vbw_edge` CLI (agent B)

```
vbw_edge --config edge/config/falcon9.cfg [--port 14650] [--bind 127.0.0.1] [--jsonl out.jsonl]
         [--log-frames] [--idle-exit SECONDS] [--max-frames N] [--quiet]
```
- One `LinkMonitor` + `BaseDetector` per `(schema_id, vid_hash)`, created on first frame; frames of other schemas
  are counted and ignored.
- Prints one line per alert rising edge: `ALERT vid=0x1A2B t=213.4 score=0.91 latency_us=37 <explanation>`.
- `--jsonl`: one JSON object per alert rising edge (`{"type":"alert","vid":"0x1a2b","t":..,"seq":..,"score":..,
  "l1_score":..,"l2_score":..,"explanation":..,"recv_ns":..}`), and with `--log-frames` one per frame
  (`{"type":"frame","vid":..,"t":..,"seq":..,"l0_ok":..,"score":..,"alert":..,"latency_us":..}`).
- Latency = steady_clock from `recvfrom` returning to the verdict being ready (decode + L0 + detector).
- On exit (idle timeout, max frames, Ctrl+C) print and append to the JSONL a summary:
  `{"type":"summary","frames":..,"vehicles":..,"alerts":..,"l0_failures":..,"latency_us":{"p50":..,"p99":..,"max":..},
    "frames_per_s":..}`.

## 6. Python side (agent A)

- `scripts/export_edge_config.py --schema falcon9` → `edge/config/falcon9.cfg` (section 2).
- `scripts/edge_golden.py` → the golden CSVs (section 3).
- `scripts/udp_replay.py`: replay one or more scenarios as UDP frames to the monitor. Options: `--scenario`
  (repeatable or `--vehicles N` copies of one scenario with distinct vehicle_ids and noise seeds), `--attack`,
  `--attack-start` (mission s), `--intensity`, `--speed` (1 = real time, 0 = as fast as possible), `--rate-hz`
  override (e.g. 50 to replay a 10 Hz flight as a 50 Hz stream), `--port`. Frames are encoded with `vbw.link.codec`
  after the attack is applied (the MITM re-encodes with a valid CRC). Writes a ground-truth JSON (vehicle_id,
  vid_hash, attack, onset t) for evaluation.
- `scripts/edge_eval.py`: builds nothing; launches `edge/build/vbw_edge.exe` with `--jsonl`, runs `udp_replay.py`,
  joins alerts with the ground truth, and writes `results/edge_eval.json` (detection per vehicle, latency from onset to
  first alert in mission seconds, per-frame processing latency p50/p99, frames/s, clean false alerts), plus the same
  scenarios through the Python `BaseDetector` for an apples-to-apples comparison.

## Phase 2 (after phase 1 passes, assigned separately)

MAVLink v2 input (`--mavlink`): decode MAVLink v2 with CRC_EXTRA validation using `edge/third_party/mavlink`, map
messages to the quad schema like `vbw/sources/mavlink_bridge.py`, and run the quad detector. L3/hardened in C++.
Linux build + Dockerfile.

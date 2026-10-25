// Telemetry wire format, mirroring vbw/link/codec.py (big-endian):
//
//   sync u32 = 0x1ACFFC1D | version u8 = 1 | schema_id u8 | vid_hash u16 | seq u16 | t f64 | n u8 |
//   n x f32 values | crc u16 (CRC-16/CCITT-FALSE over every byte before it)
//
// decode() fails (returns false) only for structurally unparseable packets, exactly where codec.decode raises
// DecodeError: too short, unknown schema / channel count, wrong length. Integrity failures (bad sync, CRC mismatch)
// are reported in L0. LinkMonitor adds the stateful checks (seq continuity mod 65536, strictly increasing t).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vbw {

constexpr uint32_t kSyncMarker = 0x1ACFFC1Du;
constexpr uint8_t kFrameVersion = 1;
constexpr size_t kHeaderSize = 19;   // sync 4 + version 1 + schema 1 + vid 2 + seq 2 + t 8 + n 1
constexpr size_t kCrcSize = 2;

constexpr size_t frame_size(size_t n_channels) { return kHeaderSize + 4 * n_channels + kCrcSize; }

// Channel count of a known schema (vbw/schema.py SCHEMAS: falcon9 = 1 -> 8, quad = 2 -> 18, spacecraft = 3 -> 1),
// or -1 if the schema id is unknown.
int expected_channels(uint8_t schema_id);

uint16_t crc16_ccitt(const uint8_t* data, size_t n, uint16_t crc = 0xFFFF);
uint16_t vehicle_hash(const std::string& vehicle_id);

struct L0 {
  bool sync_ok = true, crc_ok = true, seq_ok = true, time_ok = true;
  bool ok() const { return sync_ok && crc_ok && seq_ok && time_ok; }
};

struct Frame {
  uint8_t version = kFrameVersion, schema_id = 0;
  uint16_t vid_hash = 0, seq = 0;
  double t = 0;
  std::vector<double> values;   // float32 on the wire, widened exactly to double (Python wire_round)
};

// Parses one packet; returns false (with reason) only if structurally unparseable. Integrity failures go in l0
// (l0 is reset first; decode only ever clears sync_ok / crc_ok). `out.values` is reused, so decoding into the same
// Frame repeatedly does not allocate.
bool decode(const uint8_t* data, size_t n, Frame& out, L0& l0, std::string* reason = nullptr);

// Encodes with f.version (1 by default) and n = f.values.size() (must be <= 255). Values are narrowed to float32.
std::vector<uint8_t> encode(const Frame& f);
// Same, into a caller-owned buffer (resized to frame_size(n)).
void encode_into(const Frame& f, std::vector<uint8_t>& out);

// Ground-side stateful L0 checks, exactly like codec.LinkMonitor: seq must be last+1 (mod 65536) and t must be
// strictly greater than the previous frame's t. The previous seq/t are updated on every received frame, including
// frames that failed a check (so one dropped frame flags one seq jump, not every following frame).
class LinkMonitor {
 public:
  void reset();
  L0 receive(const Frame& f, L0 l0);

 private:
  bool has_last_ = false;
  uint16_t last_seq_ = 0;
  double last_t_ = 0;
};

}  // namespace vbw

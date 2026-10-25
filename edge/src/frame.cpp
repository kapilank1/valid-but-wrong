#include "frame.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vbw {
namespace {

constexpr std::array<uint16_t, 256> make_crc_table() {
  std::array<uint16_t, 256> table{};
  for (unsigned i = 0; i < 256; ++i) {
    uint16_t crc = static_cast<uint16_t>(i << 8);
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
    table[i] = crc;
  }
  return table;
}

constexpr std::array<uint16_t, 256> kCrcTable = make_crc_table();

inline uint16_t rd_u16(const uint8_t* p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }
inline uint32_t rd_u32(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
inline uint64_t rd_u64(const uint8_t* p) {
  return (static_cast<uint64_t>(rd_u32(p)) << 32) | static_cast<uint64_t>(rd_u32(p + 4));
}
inline void wr_u16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v >> 8);
  p[1] = static_cast<uint8_t>(v);
}
inline void wr_u32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v >> 24);
  p[1] = static_cast<uint8_t>(v >> 16);
  p[2] = static_cast<uint8_t>(v >> 8);
  p[3] = static_cast<uint8_t>(v);
}
inline void wr_u64(uint8_t* p, uint64_t v) {
  wr_u32(p, static_cast<uint32_t>(v >> 32));
  wr_u32(p + 4, static_cast<uint32_t>(v));
}

// double -> float32 like struct.pack('>f'): round to nearest; out-of-range finite values become +-inf instead of
// raising (Python raises OverflowError; we never encode such values in practice).
inline float narrow(double v) {
  constexpr double kMax = static_cast<double>(std::numeric_limits<float>::max());
  if (v > kMax || v < -kMax) {   // includes +-inf; NaN falls through to the cast (stays a quiet NaN)
    // IEEE round-to-nearest-even: from FLT_MAX + half an ulp upward the result is inf, below it FLT_MAX.
    // Spelled out because converting an out-of-range double to float is formally undefined in C++.
    constexpr double kOverflow = 3.4028235677973366e38;
    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kFltMax = std::numeric_limits<float>::max();
    if (std::fabs(v) >= kOverflow) return v > 0 ? kInf : -kInf;
    return v > 0 ? kFltMax : -kFltMax;
  }
  return static_cast<float>(v);
}

}  // namespace

int expected_channels(uint8_t schema_id) {
  switch (schema_id) {
    case 1: return 8;    // falcon9
    case 2: return 18;   // quad
    case 3: return 1;    // spacecraft
    default: return -1;
  }
}

uint16_t crc16_ccitt(const uint8_t* data, size_t n, uint16_t crc) {
  for (size_t i = 0; i < n; ++i) {
    crc = static_cast<uint16_t>((crc << 8) ^ kCrcTable[((crc >> 8) ^ data[i]) & 0xFF]);
  }
  return crc;
}

uint16_t vehicle_hash(const std::string& vehicle_id) {
  return crc16_ccitt(reinterpret_cast<const uint8_t*>(vehicle_id.data()), vehicle_id.size());
}

bool decode(const uint8_t* data, size_t n, Frame& out, L0& l0, std::string* reason) {
  l0 = L0{};
  if (n < kHeaderSize + kCrcSize) {
    if (reason) *reason = "packet too short: " + std::to_string(n) + " bytes";
    return false;
  }
  const uint32_t sync = rd_u32(data);
  const uint8_t schema_id = data[5];
  const uint8_t nch = data[18];
  const int expected = expected_channels(schema_id);
  if (expected < 0 || nch != expected) {
    if (reason) {
      *reason = "unknown schema " + std::to_string(schema_id) + " with " + std::to_string(nch) + " channels";
    }
    return false;
  }
  const size_t expected_len = frame_size(nch);
  if (n != expected_len) {
    if (reason) *reason = "length " + std::to_string(n) + " != " + std::to_string(expected_len);
    return false;
  }

  out.version = data[4];
  out.schema_id = schema_id;
  out.vid_hash = rd_u16(data + 6);
  out.seq = rd_u16(data + 8);
  const uint64_t tbits = rd_u64(data + 10);
  std::memcpy(&out.t, &tbits, sizeof(double));
  out.values.resize(nch);
  const uint8_t* p = data + kHeaderSize;
  for (size_t i = 0; i < nch; ++i, p += 4) {
    const uint32_t bits = rd_u32(p);
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    out.values[i] = static_cast<double>(f);
  }

  if (sync != kSyncMarker) l0.sync_ok = false;
  const uint16_t crc = rd_u16(data + expected_len - kCrcSize);
  if (crc != crc16_ccitt(data, expected_len - kCrcSize)) l0.crc_ok = false;
  return true;
}

void encode_into(const Frame& f, std::vector<uint8_t>& out) {
  if (f.values.size() > 255) throw std::invalid_argument("encode: more than 255 channels");
  const size_t nch = f.values.size();
  out.resize(frame_size(nch));
  uint8_t* p = out.data();
  wr_u32(p, kSyncMarker);
  p[4] = f.version;
  p[5] = f.schema_id;
  wr_u16(p + 6, f.vid_hash);
  wr_u16(p + 8, f.seq);
  uint64_t tbits;
  std::memcpy(&tbits, &f.t, sizeof(double));
  wr_u64(p + 10, tbits);
  p[18] = static_cast<uint8_t>(nch);
  uint8_t* q = p + kHeaderSize;
  for (size_t i = 0; i < nch; ++i, q += 4) {
    const float v = narrow(f.values[i]);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(float));
    wr_u32(q, bits);
  }
  const size_t body = out.size() - kCrcSize;
  wr_u16(p + body, crc16_ccitt(p, body));
}

std::vector<uint8_t> encode(const Frame& f) {
  std::vector<uint8_t> out;
  encode_into(f, out);
  return out;
}

void LinkMonitor::reset() {
  has_last_ = false;
  last_seq_ = 0;
  last_t_ = 0;
}

L0 LinkMonitor::receive(const Frame& f, L0 l0) {
  if (has_last_) {
    if (f.seq != static_cast<uint16_t>(last_seq_ + 1)) l0.seq_ok = false;
    if (f.t <= last_t_) l0.time_ok = false;   // NaN compares false, exactly as in Python
  }
  has_last_ = true;
  last_seq_ = f.seq;
  last_t_ = f.t;
  return l0;
}

}  // namespace vbw

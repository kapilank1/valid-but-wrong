// Wire format, CRC, LinkMonitor and UDP loopback tests. Reference packets were produced by the Python codec:
//   uv run python -c "from vbw.link.codec import encode; ..."   (vbw/link/codec.py)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "frame.hpp"
#include "udp.hpp"

using namespace vbw;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
    }                                                                            \
  } while (0)

static std::vector<uint8_t> from_hex(const char* hex) {
  std::vector<uint8_t> out;
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; hex[i] && hex[i + 1]; i += 2) out.push_back(static_cast<uint8_t>(nib(hex[i]) * 16 + nib(hex[i + 1])));
  return out;
}

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// falcon9:ses-9, seq 1234, t 190.1, [2345.678, 51234.5, 812.25, 2201.125, 30123.0, 21.3, 33.7, 12345.6]
static const char* kP1 =
    "1acffc1d01016fb804d24067c333333333330845129ad947482280444b10004509920046eb560041aa66664206cccd4640e666f9a6";
// falcon9:ses-9, seq 65535, t 0.1, [0.0, -1.5, 1e-7, 3.4e38, nan, inf, -0.0, 1.0]
static const char* kP2 =
    "1acffc1d01016fb8ffff3fb999999999999a0800000000bfc0000033d6bf957f7fc99e7fc000007f800000800000003f800000a2b0";
// quad-1 (schema 2, 18 channels), seq 7, t 12.345678901234567, values i*0.5-3
static const char* kP3 =
    "1acffc1d01020ef800074028b0fcd32f707a12c0400000c0200000c0000000bfc00000bf800000bf000000000000003f0000003f800000"
    "3fc0000040000000402000004040000040600000408000004090000040a0000040b00000ec2e";
// sc (schema 3, 1 channel), seq 0, t -5.0, [0.333333333]
static const char* kP4 = "1acffc1d01031cc00000c014000000000000013eaaaaabca2e";

static void test_crc() {
  const char* check = "123456789";
  CHECK(crc16_ccitt(reinterpret_cast<const uint8_t*>(check), 9) == 0x29B1);
  CHECK(crc16_ccitt(nullptr, 0) == 0xFFFF);
  // Incremental CRC equals one-shot CRC.
  uint16_t c = crc16_ccitt(reinterpret_cast<const uint8_t*>(check), 4);
  c = crc16_ccitt(reinterpret_cast<const uint8_t*>(check) + 4, 5, c);
  CHECK(c == 0x29B1);
  // vehicle_hash values from vbw.link.codec.vehicle_hash
  CHECK(vehicle_hash("falcon9:ses-9") == 0x6FB8);
  CHECK(vehicle_hash("quad-1") == 0x0EF8);
  CHECK(vehicle_hash("") == 0xFFFF);
  CHECK(vehicle_hash("falcon9:ses-9#3") == 0x6842);
  CHECK(vehicle_hash("sc") == 0x1CC0);
}

static void test_decode_python_packets() {
  Frame f;
  L0 l0;
  std::string reason;

  auto p1 = from_hex(kP1);
  CHECK(p1.size() == 53);
  CHECK(decode(p1.data(), p1.size(), f, l0, &reason));
  CHECK(l0.ok());
  CHECK(f.version == 1);
  CHECK(f.schema_id == 1);
  CHECK(f.vid_hash == 0x6FB8);
  CHECK(f.seq == 1234);
  CHECK(f.t == 190.1);
  const double v1[8] = {2345.677978515625, 51234.5, 812.25, 2201.125, 30123.0,
                        21.299999237060547, 33.70000076293945, 12345.599609375};
  CHECK(f.values.size() == 8);
  for (size_t i = 0; i < 8 && i < f.values.size(); ++i) CHECK(same_bits(f.values[i], v1[i]));
  CHECK(encode(f) == p1);

  auto p2 = from_hex(kP2);
  CHECK(decode(p2.data(), p2.size(), f, l0, &reason));
  CHECK(l0.ok());
  CHECK(f.seq == 65535);
  CHECK(f.t == 0.1);
  CHECK(f.values.size() == 8);
  if (f.values.size() == 8) {
    CHECK(same_bits(f.values[0], 0.0));
    CHECK(f.values[1] == -1.5);
    CHECK(same_bits(f.values[2], 1.0000000116860974e-07));
    CHECK(same_bits(f.values[3], 3.3999999521443642e+38));
    CHECK(std::isnan(f.values[4]));
    CHECK(std::isinf(f.values[5]) && f.values[5] > 0);
    CHECK(f.values[6] == 0.0 && std::signbit(f.values[6]));
    CHECK(f.values[7] == 1.0);
  }
  CHECK(encode(f) == p2);   // NaN stays the canonical quiet NaN 0x7fc00000

  auto p3 = from_hex(kP3);
  CHECK(p3.size() == 93);
  CHECK(decode(p3.data(), p3.size(), f, l0, &reason));
  CHECK(l0.ok());
  CHECK(f.schema_id == 2 && f.vid_hash == 0x0EF8 && f.seq == 7);
  CHECK(f.t == 12.345678901234567);
  CHECK(f.values.size() == 18);
  for (size_t i = 0; i < f.values.size(); ++i) CHECK(f.values[i] == static_cast<double>(i) * 0.5 - 3.0);
  CHECK(encode(f) == p3);

  auto p4 = from_hex(kP4);
  CHECK(p4.size() == 25);
  CHECK(decode(p4.data(), p4.size(), f, l0, &reason));
  CHECK(l0.ok());
  CHECK(f.schema_id == 3 && f.vid_hash == 0x1CC0 && f.seq == 0 && f.t == -5.0);
  CHECK(f.values.size() == 1 && f.values[0] == static_cast<double>(0.333333333f));
  CHECK(encode(f) == p4);
}

static void test_round_trip() {
  Frame f;
  f.schema_id = 1;
  f.vid_hash = vehicle_hash("falcon9:crs-20");
  f.seq = 42;
  f.t = 123.456789012345;
  f.values = {1.1, -2.2, 3.3e3, 4.4e-3, 5.5e6, 6.6, -7.7, 8.8e4};
  auto bytes = encode(f);
  CHECK(bytes.size() == frame_size(8));
  Frame g;
  L0 l0;
  CHECK(decode(bytes.data(), bytes.size(), g, l0));
  CHECK(l0.ok());
  CHECK(g.schema_id == f.schema_id && g.vid_hash == f.vid_hash && g.seq == f.seq && same_bits(g.t, f.t));
  for (size_t i = 0; i < 8; ++i) CHECK(g.values[i] == static_cast<double>(static_cast<float>(f.values[i])));
  // Re-encoding the decoded (already float32) frame is byte-identical.
  CHECK(encode(g) == bytes);
  // Float32 overflow saturates like IEEE narrowing.
  Frame big = f;
  big.values[0] = 1e300;
  big.values[1] = -1e300;
  big.values[2] = 3.4028235e38;   // rounds to FLT_MAX
  auto bb = encode(big);
  CHECK(decode(bb.data(), bb.size(), g, l0));
  CHECK(std::isinf(g.values[0]) && g.values[0] > 0);
  CHECK(std::isinf(g.values[1]) && g.values[1] < 0);
  CHECK(g.values[2] == static_cast<double>(std::numeric_limits<float>::max()));
}

static void test_corruption() {
  Frame f;
  L0 l0;
  std::string reason;
  const auto p1 = from_hex(kP1);

  // Bit flip in a value: parses, CRC fails, sync fine.
  auto bad = p1;
  bad[25] ^= 0x01;
  CHECK(decode(bad.data(), bad.size(), f, l0, &reason));
  CHECK(l0.sync_ok && !l0.crc_ok && !l0.ok());
  CHECK(l0.seq_ok && l0.time_ok);

  // Corrupted CRC field itself.
  bad = p1;
  bad[bad.size() - 1] ^= 0xFF;
  CHECK(decode(bad.data(), bad.size(), f, l0, &reason));
  CHECK(l0.sync_ok && !l0.crc_ok);

  // Bad sync: both sync and CRC fail (the CRC covers the sync word), as in Python.
  bad = p1;
  bad[0] = 0x00;
  CHECK(decode(bad.data(), bad.size(), f, l0, &reason));
  CHECK(!l0.sync_ok && !l0.crc_ok);

  // Bad sync with a recomputed (valid) CRC: only sync fails.
  const uint16_t crc = crc16_ccitt(bad.data(), bad.size() - 2);
  bad[bad.size() - 2] = static_cast<uint8_t>(crc >> 8);
  bad[bad.size() - 1] = static_cast<uint8_t>(crc);
  CHECK(decode(bad.data(), bad.size(), f, l0, &reason));
  CHECK(!l0.sync_ok && l0.crc_ok);

  // decode resets l0 each call.
  CHECK(decode(p1.data(), p1.size(), f, l0, &reason));
  CHECK(l0.ok());

  // Structurally unparseable: too short, wrong length, unknown schema, wrong channel count.
  reason.clear();
  CHECK(!decode(p1.data(), 20, f, l0, &reason));
  CHECK(reason == "packet too short: 20 bytes");
  auto longer = p1;
  longer.push_back(0);
  CHECK(!decode(longer.data(), longer.size(), f, l0, &reason));
  CHECK(reason == "length 54 != 53");
  CHECK(!decode(p1.data(), p1.size() - 1, f, l0, &reason));
  CHECK(reason == "length 52 != 53");
  auto unk = p1;
  unk[5] = 9;
  CHECK(!decode(unk.data(), unk.size(), f, l0, &reason));
  CHECK(reason == "unknown schema 9 with 8 channels");
  auto badn = p1;
  badn[18] = 7;
  CHECK(!decode(badn.data(), badn.size(), f, l0, &reason));
  CHECK(reason == "unknown schema 1 with 7 channels");
  CHECK(!decode(nullptr, 0, f, l0));   // reason pointer optional
}

static Frame mk(uint16_t seq, double t) {
  Frame f;
  f.schema_id = 1;
  f.seq = seq;
  f.t = t;
  f.values.assign(8, 0.0);
  return f;
}

static void test_link_monitor() {
  LinkMonitor m;
  // First frame is always continuous; wrap 65535 -> 0 is continuous.
  CHECK(m.receive(mk(65534, 1.0), L0{}).ok());
  CHECK(m.receive(mk(65535, 1.1), L0{}).ok());
  CHECK(m.receive(mk(0, 1.2), L0{}).ok());
  CHECK(m.receive(mk(1, 1.3), L0{}).ok());
  // Seq jump flags exactly one frame (last seq is updated on every frame).
  L0 r = m.receive(mk(5, 1.4), L0{});
  CHECK(!r.seq_ok && r.time_ok && r.sync_ok && r.crc_ok);
  CHECK(m.receive(mk(6, 1.5), L0{}).ok());
  // Repeated seq is a jump too.
  r = m.receive(mk(6, 1.6), L0{});
  CHECK(!r.seq_ok && r.time_ok);
  CHECK(m.receive(mk(7, 1.7), L0{}).ok());
  // Non-monotonic t: equal and decreasing both fail; last t follows the received frame (Python semantics), so a
  // frame after a backwards step only has to beat the backwards t.
  r = m.receive(mk(8, 1.7), L0{});
  CHECK(r.seq_ok && !r.time_ok);
  r = m.receive(mk(9, 1.0), L0{});
  CHECK(r.seq_ok && !r.time_ok);
  CHECK(m.receive(mk(10, 1.05), L0{}).ok());
  // Incoming sync/CRC flags are preserved.
  L0 in;
  in.crc_ok = false;
  r = m.receive(mk(11, 2.0), in);
  CHECK(!r.crc_ok && r.seq_ok && r.time_ok && r.sync_ok);
  // NaN t compares false: never flagged, and the frame after it is not flagged either.
  CHECK(m.receive(mk(12, std::numeric_limits<double>::quiet_NaN()), L0{}).time_ok);
  CHECK(m.receive(mk(13, 0.5), L0{}).time_ok);
  // reset() forgets history.
  m.reset();
  CHECK(m.receive(mk(100, 0.0), L0{}).ok());
  r = m.receive(mk(100, 0.0), L0{});
  CHECK(!r.seq_ok && !r.time_ok);
}

static void test_udp_loopback() {
  try {
    UdpReceiver rx("127.0.0.1", 0);
    CHECK(rx.port() != 0);
    std::printf("udp: bound 127.0.0.1:%u, SO_RCVBUF = %d bytes\n", rx.port(), rx.rcvbuf_bytes());
    CHECK(rx.rcvbuf_bytes() >= 4 * 1024 * 1024);
    rx.set_timeout_ms(50);
    uint8_t buf[65536];
    size_t len = 0;
    CHECK(rx.receive(buf, sizeof buf, len) == UdpReceiver::Status::Timeout);

    UdpSender tx("127.0.0.1", rx.port());
    const auto p1 = from_hex(kP1);
    const auto p3 = from_hex(kP3);
    CHECK(tx.send(p1.data(), p1.size()));
    CHECK(tx.send(p3.data(), p3.size()));
    rx.set_timeout_ms(2000);
    CHECK(rx.receive(buf, sizeof buf, len) == UdpReceiver::Status::Packet);
    CHECK(len == p1.size() && std::memcmp(buf, p1.data(), len) == 0);
    Frame f;
    L0 l0;
    CHECK(decode(buf, len, f, l0) && l0.ok() && f.seq == 1234);
    CHECK(rx.receive(buf, sizeof buf, len) == UdpReceiver::Status::Packet);
    CHECK(len == p3.size() && !rx.truncated());
    // Truncation is reported.
    CHECK(tx.send(p1.data(), p1.size()));
    CHECK(rx.receive(buf, 10, len) == UdpReceiver::Status::Packet);
    CHECK(len == 10 && rx.truncated());
    // Binding the same port twice fails with an exception that names the port.
    bool threw = false;
    try {
      UdpReceiver rx2("127.0.0.1", rx.port());
    } catch (const std::exception& e) {
      threw = std::string(e.what()).find(std::to_string(rx.port())) != std::string::npos;
    }
    CHECK(threw);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL udp loopback threw: %s\n", e.what());
    ++g_failures;
  }
}

int main() {
  test_crc();
  test_decode_python_packets();
  test_round_trip();
  test_corruption();
  test_link_monitor();
  test_udp_loopback();
  if (g_failures) {
    std::fprintf(stderr, "test_frame: %d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_frame: all %d checks passed\n", g_checks);
  return 0;
}

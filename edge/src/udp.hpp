// Minimal UDP sockets for the edge monitor: Winsock on Windows (WSAStartup/WSACleanup via RAII), POSIX elsewhere.
//
//   UdpReceiver rx("127.0.0.1", 14650);          // binds, SO_RCVBUF >= 4 MB requested
//   rx.set_timeout_ms(200);                      // receive() returns Timeout after this long without a datagram
//   size_t len; if (rx.receive(buf, sizeof buf, len) == UdpReceiver::Status::Packet) { ... }
//
// The timeout is a socket option (SO_RCVTIMEO), so a receive costs one system call; the main loop uses the
// timeouts to check for Ctrl+C and idle exit. Errors in construction throw std::runtime_error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vbw {

// Keeps the platform socket library initialised while alive (Winsock is reference counted; no-op on POSIX).
class NetInit {
 public:
  NetInit();
  ~NetInit();
  NetInit(const NetInit&) = delete;
  NetInit& operator=(const NetInit&) = delete;
};

class UdpReceiver {
 public:
  enum class Status { Packet, Timeout, Error };

  // host: IPv4 address or name ("0.0.0.0" for all interfaces); port 0 picks an ephemeral port (see port()).
  UdpReceiver(const std::string& host, uint16_t port, int rcvbuf_bytes = 4 * 1024 * 1024);
  ~UdpReceiver();
  UdpReceiver(const UdpReceiver&) = delete;
  UdpReceiver& operator=(const UdpReceiver&) = delete;

  void set_timeout_ms(int ms);   // 0 = block forever
  // Receives one datagram into buf. A datagram longer than cap is truncated to cap bytes (Status::Packet,
  // truncated() == true). Interrupted calls and Windows' spurious UDP WSAECONNRESET are reported as Timeout.
  Status receive(uint8_t* buf, size_t cap, size_t& len);

  bool truncated() const { return truncated_; }
  uint16_t port() const { return port_; }             // the bound port
  const std::string& host() const { return host_; }
  int rcvbuf_bytes() const { return rcvbuf_; }        // SO_RCVBUF as reported by the OS after setting it
  const std::string& last_error() const { return error_; }

 private:
  NetInit net_;
  intptr_t sock_ = -1;
  std::string host_;
  uint16_t port_ = 0;
  int rcvbuf_ = 0;
  int timeout_ms_ = -1;
  bool truncated_ = false;
  std::string error_;
};

// Sender for tests and tools (loopback replay); not used on the monitor's hot path.
class UdpSender {
 public:
  UdpSender(const std::string& host, uint16_t port);
  ~UdpSender();
  UdpSender(const UdpSender&) = delete;
  UdpSender& operator=(const UdpSender&) = delete;
  bool send(const uint8_t* data, size_t n);

 private:
  NetInit net_;
  intptr_t sock_ = -1;
  unsigned char addr_[16] = {};   // sockaddr_in
};

}  // namespace vbw

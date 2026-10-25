#include "udp.hpp"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
#endif
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace vbw {
namespace {

#ifdef _WIN32
using sock_t = SOCKET;
const sock_t kBad = INVALID_SOCKET;
inline sock_t S(intptr_t s) { return static_cast<sock_t>(s); }
int last_net_error() { return WSAGetLastError(); }
std::string net_error_text(int code) {
  char buf[256] = {};
  DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                           static_cast<DWORD>(code), 0, buf, sizeof(buf), nullptr);
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ' || buf[n - 1] == '.')) buf[--n] = 0;
  return std::string(buf) + " (WSA error " + std::to_string(code) + ")";
}
void close_socket(intptr_t s) { closesocket(S(s)); }
#else
using sock_t = int;
const sock_t kBad = -1;
inline sock_t S(intptr_t s) { return static_cast<sock_t>(s); }
int last_net_error() { return errno; }
std::string net_error_text(int code) { return std::string(std::strerror(code)) + " (errno " + std::to_string(code) + ")"; }
void close_socket(intptr_t s) { ::close(S(s)); }
#endif

sockaddr_in resolve_ipv4(const std::string& host, uint16_t port, bool passive) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;
  if (passive) hints.ai_flags = AI_PASSIVE;
  addrinfo* res = nullptr;
  const char* node = host.empty() ? nullptr : host.c_str();
  const int rc = getaddrinfo(node, nullptr, &hints, &res);
  if (rc != 0 || res == nullptr) {
    throw std::runtime_error("udp: cannot resolve host '" + host + "' (getaddrinfo error " + std::to_string(rc) + ")");
  }
  sockaddr_in addr{};
  std::memcpy(&addr, res->ai_addr, sizeof(addr));
  freeaddrinfo(res);
  addr.sin_port = htons(port);
  return addr;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ NetInit

NetInit::NetInit() {
#ifdef _WIN32
  WSADATA wsa;
  const int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
  if (rc != 0) throw std::runtime_error("udp: WSAStartup failed: " + net_error_text(rc));
#endif
}

NetInit::~NetInit() {
#ifdef _WIN32
  WSACleanup();
#endif
}

// ------------------------------------------------------------------------------------------------ UdpReceiver

UdpReceiver::UdpReceiver(const std::string& host, uint16_t port, int rcvbuf_bytes) : host_(host) {
  const sockaddr_in addr = resolve_ipv4(host, port, true);
  sock_t s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s == kBad) throw std::runtime_error("udp: socket() failed: " + net_error_text(last_net_error()));
  sock_ = static_cast<intptr_t>(s);

  if (rcvbuf_bytes > 0) {
    int want = rcvbuf_bytes;
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&want), sizeof(want));
  }
  {
    int got = 0;
    socklen_t len = sizeof(got);
    if (getsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<char*>(&got), &len) == 0) rcvbuf_ = got;
  }

#ifdef _WIN32
  // A UDP socket on Windows can report WSAECONNRESET from an ICMP "port unreachable"; it is meaningless for a
  // receiver, so turn it off.
  BOOL off = FALSE;
  DWORD bytes = 0;
  WSAIoctl(s, SIO_UDP_CONNRESET, &off, sizeof(off), nullptr, 0, &bytes, nullptr, nullptr);
#endif

  if (::bind(s, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
    const int err = last_net_error();
    close_socket(sock_);
    sock_ = -1;
    throw std::runtime_error("udp: cannot bind " + host + ":" + std::to_string(port) + ": " + net_error_text(err));
  }
  sockaddr_in bound{};
  socklen_t blen = sizeof(bound);
  if (getsockname(s, reinterpret_cast<sockaddr*>(&bound), &blen) == 0) {
    port_ = ntohs(bound.sin_port);
  } else {
    port_ = port;
  }
}

UdpReceiver::~UdpReceiver() {
  if (sock_ != -1) close_socket(sock_);
}

void UdpReceiver::set_timeout_ms(int ms) {
  if (ms < 0) ms = 0;
  if (ms == timeout_ms_) return;
  timeout_ms_ = ms;
#ifdef _WIN32
  DWORD tv = static_cast<DWORD>(ms);
  setsockopt(S(sock_), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
  timeval tv{};
  tv.tv_sec = ms / 1000;
  tv.tv_usec = (ms % 1000) * 1000;
  setsockopt(S(sock_), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

UdpReceiver::Status UdpReceiver::receive(uint8_t* buf, size_t cap, size_t& len) {
  truncated_ = false;
  len = 0;
#ifdef _WIN32
  const int n = ::recvfrom(S(sock_), reinterpret_cast<char*>(buf), static_cast<int>(cap), 0, nullptr, nullptr);
  if (n >= 0) {
    len = static_cast<size_t>(n);
    return Status::Packet;
  }
  const int err = WSAGetLastError();
  if (err == WSAETIMEDOUT || err == WSAEINTR || err == WSAECONNRESET || err == WSAEWOULDBLOCK) return Status::Timeout;
  if (err == WSAEMSGSIZE) {
    truncated_ = true;
    len = cap;
    return Status::Packet;
  }
  error_ = net_error_text(err);
  return Status::Error;
#else
#ifdef __linux__
  const int flags = MSG_TRUNC;   // return the real datagram length so truncation is detectable
#else
  const int flags = 0;
#endif
  const ssize_t n = ::recvfrom(S(sock_), buf, cap, flags, nullptr, nullptr);
  if (n >= 0) {
    if (static_cast<size_t>(n) > cap) {
      truncated_ = true;
      len = cap;
    } else {
      len = static_cast<size_t>(n);
    }
    return Status::Packet;
  }
  const int err = errno;
  if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR || err == ECONNREFUSED) return Status::Timeout;
  error_ = net_error_text(err);
  return Status::Error;
#endif
}

// ------------------------------------------------------------------------------------------------ UdpSender

UdpSender::UdpSender(const std::string& host, uint16_t port) {
  static_assert(sizeof(addr_) >= sizeof(sockaddr_in), "addr_ too small");
  const sockaddr_in addr = resolve_ipv4(host, port, false);
  std::memcpy(addr_, &addr, sizeof(addr));
  sock_t s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s == kBad) throw std::runtime_error("udp: socket() failed: " + net_error_text(last_net_error()));
  sock_ = static_cast<intptr_t>(s);
}

UdpSender::~UdpSender() {
  if (sock_ != -1) close_socket(sock_);
}

bool UdpSender::send(const uint8_t* data, size_t n) {
#ifdef _WIN32
  const int rc = ::sendto(S(sock_), reinterpret_cast<const char*>(data), static_cast<int>(n), 0,
                          reinterpret_cast<const sockaddr*>(addr_), static_cast<int>(sizeof(sockaddr_in)));
#else
  const ssize_t rc = ::sendto(S(sock_), data, n, 0, reinterpret_cast<const sockaddr*>(addr_), sizeof(sockaddr_in));
#endif
  return rc >= 0 && static_cast<size_t>(rc) == n;
}

}  // namespace vbw

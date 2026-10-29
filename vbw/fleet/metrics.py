"""Minimal Prometheus metrics: counters, gauges and histograms rendered in the text exposition format.

Latency is a histogram, not a p99 gauge: bucket counts from many pods can be summed and a fleet-wide percentile
computed from the sum, whereas per-pod percentiles can't be combined into a fleet percentile.
"""

from __future__ import annotations

import bisect
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


class Counter:
    def __init__(self, name: str, help: str):
        self.name, self.help, self.value = name, help, 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def render(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter", f"{self.name} {self.value:g}"]


class Gauge(Counter):
    def set(self, value: float) -> None:
        self.value = value

    def render(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge", f"{self.name} {self.value:g}"]


class Histogram:
    def __init__(self, name: str, help: str, buckets: tuple[float, ...]):
        self.name, self.help = name, help
        self.buckets = tuple(sorted(buckets))
        self.counts = [0] * (len(self.buckets) + 1)   # last slot is +Inf
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, value)] += 1
        self.sum += value
        self.count += 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cumulative = 0
        for bound, n in zip((*self.buckets, float("inf")), self.counts):
            cumulative += n
            le = "+Inf" if bound == float("inf") else f"{bound:g}"
            lines.append(f'{self.name}_bucket{{le="{le}"}} {cumulative}')
        lines += [f"{self.name}_sum {self.sum:g}", f"{self.name}_count {self.count}"]
        return lines


def render(metrics: list[Counter | Histogram]) -> str:
    return "\n".join(line for m in metrics for line in m.render()) + "\n"


def serve(port: int, metrics: Callable[[], str], ready: Callable[[], bool]) -> ThreadingHTTPServer:
    """Serve /metrics and /healthz on a daemon thread. /healthz is 503 until ready() is true."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/metrics":
                self._send(200, metrics(), "text/plain; version=0.0.4")
            elif self.path == "/healthz":
                ok = ready()
                self._send(200 if ok else 503, "ok\n" if ok else "starting\n", "text/plain")
            else:
                self._send(404, "not found\n", "text/plain")

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:   # probes hit this every few seconds; keep the logs readable
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

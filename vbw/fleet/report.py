"""Summarize fleet workers from two /metrics scrapes taken `seconds` apart (used by scripts/fleet_report.py).

Rates come from counter deltas; fleet-wide latency percentiles come from the summed histogram buckets, the same
way Prometheus' histogram_quantile() does it.
"""

from __future__ import annotations

from dataclasses import dataclass

Samples = dict[str, float]


def parse(text: str) -> Samples:
    """Prometheus text format -> {'name' or 'name{labels}': value}. Comments and blank lines are skipped."""
    out: Samples = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.rpartition(" ")
            out[key] = float(value)
    return out


def buckets(samples: Samples, name: str) -> list[tuple[float, float]]:
    """Cumulative (upper bound, count) pairs of a histogram, sorted by bound."""
    prefix = f'{name}_bucket{{le="'
    pairs = [(float(k[len(prefix):-2]), v) for k, v in samples.items() if k.startswith(prefix)]
    return sorted(pairs)


def quantile(q: float, cumulative: list[tuple[float, float]]) -> float | None:
    """Linear interpolation inside the bucket that holds the q-th observation (histogram_quantile semantics)."""
    if not cumulative or cumulative[-1][1] == 0:
        return None
    rank = q * cumulative[-1][1]
    lower_bound, lower_count = 0.0, 0.0
    for bound, count in cumulative:
        if count >= rank:
            if bound == float("inf"):
                return lower_bound          # can't interpolate into +Inf; report the last finite bound
            width = count - lower_count
            return lower_bound + (bound - lower_bound) * ((rank - lower_count) / width if width else 0.0)
        lower_bound, lower_count = bound, count
    return None


def delta_buckets(before: Samples, after: Samples, name: str) -> list[tuple[float, float]]:
    old = dict(buckets(before, name))
    return [(bound, count - old.get(bound, 0.0)) for bound, count in buckets(after, name)]


@dataclass
class PodSummary:
    pod: str
    vehicles: int
    target_fps: float
    fps: float
    lag_s: float
    lag_trend: float           # change in lag per second over the window: < 0 means it is catching up


@dataclass
class FleetSummary:
    pods: list[PodSummary]
    p50_ms: float | None
    p99_ms: float | None
    flights_attacked: int
    flights_detected: int
    flights_clean: int
    false_alerts: int
    detection_p50_s: float | None

    @property
    def vehicles(self) -> int:
        return sum(p.vehicles for p in self.pods)

    @property
    def fps(self) -> float:
        return sum(p.fps for p in self.pods)

    @property
    def target_fps(self) -> float:
        return sum(p.target_fps for p in self.pods)

    @property
    def keeping_up(self) -> bool:
        """Every pod is either at real time or draining its backlog (e.g. just after a scale-up)."""
        return all(p.lag_s < 1.0 or p.lag_trend < 0 for p in self.pods)


def summarize(before: dict[str, Samples], after: dict[str, Samples], seconds: float) -> FleetSummary:
    pods, latency, detection = [], {}, {}
    for pod in sorted(after):
        a, b = after[pod], before.get(pod, {})
        pods.append(PodSummary(
            pod=pod,
            vehicles=int(a.get("fleet_vehicles", 0)),
            target_fps=a.get("fleet_target_frames_per_second", 0.0),
            fps=(a.get("fleet_frames_total", 0.0) - b.get("fleet_frames_total", 0.0)) / seconds,
            lag_s=a.get("fleet_lag_seconds", 0.0),
            lag_trend=(a.get("fleet_lag_seconds", 0.0) - b.get("fleet_lag_seconds", 0.0)) / seconds,
        ))
        for bound, count in delta_buckets(b, a, "fleet_frame_seconds"):
            latency[bound] = latency.get(bound, 0.0) + count
        for bound, count in buckets(a, "fleet_detection_latency_seconds"):
            detection[bound] = detection.get(bound, 0.0) + count

    def total(name: str) -> int:
        return int(sum(s.get(name, 0.0) for s in after.values()))

    lat = sorted(latency.items())
    p50, p99 = quantile(0.5, lat), quantile(0.99, lat)
    return FleetSummary(
        pods=pods,
        p50_ms=None if p50 is None else p50 * 1e3,
        p99_ms=None if p99 is None else p99 * 1e3,
        flights_attacked=total("fleet_flights_attacked_total"),
        flights_detected=total("fleet_flights_detected_total"),
        flights_clean=total("fleet_flights_clean_total"),
        false_alerts=total("fleet_flights_false_alert_total"),
        detection_p50_s=quantile(0.5, sorted(detection.items())),
    )

"""One detector worker: owns a shard of the fleet and scores every frame of its vehicles in real time.

Each tick processes one frame per owned vehicle, and ticks are paced at 10 Hz x FLEET_SPEED. If a tick takes
longer than the frame period the worker falls behind real time; fleet_lag_seconds is how far behind, which is the
saturation signal: a healthy shard holds it near 0, an overloaded one grows it without bound.

Configuration (environment):
    FLEET_SIZE             vehicles in the whole fleet (default 200)
    FLEET_ATTACK_FRACTION  share of vehicles under a scripted MITM attack (default 0.1)
    FLEET_SPEED            1.0 = real time, 10 Hz per vehicle (default 1.0)
    FLEET_LAUNCHES         "holdout" (6 launches no model trained on; default) or "all" (31)
    FLEET_DETECTOR         "base" (L1+L2, default) or "hardened" (+L3)
    FLEET_WORKER_SELECTOR  label selector of the worker pods; when set and running in a pod, vehicles are sharded
                           over the Ready ones (read from the API every 2 s)
    FLEET_REPLICAS         outside a cluster: shard over ordinals 0..FLEET_REPLICAS-1 (default 1)
    FLEET_ORDINAL          this worker's shard; default parsed from HOSTNAME (worker-3 -> 3)
    METRICS_PORT           /metrics and /healthz (default 9100)
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable

from vbw.detector.splits import all_slugs, holdout_slugs
from vbw.fleet import k8s
from vbw.fleet.metrics import Counter, Gauge, Histogram, render, serve
from vbw.fleet.sharding import ordinal_from_hostname, owner
from vbw.fleet.vehicle import FlightResult, VehicleStream, fleet_specs
from vbw.schema import FALCON9, Detector
from vbw.sources.falcon9 import RATE_HZ

log = logging.getLogger("vbw.fleet")

FRAME_LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1)
DETECTION_LATENCY_BUCKETS = (1, 2, 5, 10, 20, 30, 60, 120, 300)


@dataclass
class WorkerConfig:
    ordinal: int
    fleet_size: int = 200
    attack_fraction: float = 0.1
    speed: float = 1.0
    launches: str = "holdout"
    detector: str = "base"
    worker_selector: str | None = None
    replicas: int = 1
    metrics_port: int = 9100
    reshard_every_s: float = 2.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "WorkerConfig":
        env = dict(os.environ) if env is None else env
        ordinal = env.get("FLEET_ORDINAL")
        return cls(
            ordinal=int(ordinal) if ordinal else ordinal_from_hostname(env.get("HOSTNAME") or socket.gethostname()),
            fleet_size=int(env.get("FLEET_SIZE", 200)),
            attack_fraction=float(env.get("FLEET_ATTACK_FRACTION", 0.1)),
            speed=float(env.get("FLEET_SPEED", 1.0)),
            launches=env.get("FLEET_LAUNCHES", "holdout"),
            detector=env.get("FLEET_DETECTOR", "base"),
            worker_selector=env.get("FLEET_WORKER_SELECTOR") or None,
            replicas=int(env.get("FLEET_REPLICAS", 1)),
            metrics_port=int(env.get("METRICS_PORT", 9100)),
        )


def detector_factory(name: str) -> Callable[[], Detector]:
    """Per-vehicle detectors that share the read-only parts (ONNX session, thresholds) instead of loading them
    once per vehicle: a few hundred ONNX Runtime sessions per pod would cost far more memory than the state."""
    from vbw.detector.alerting import AlertConfig
    from vbw.detector.base import BaseDetector
    from vbw.detector.l2 import L2Model
    from vbw.detector.physics import L1Config

    if name == "hardened":
        from vbw.detector.l3 import HardenedDetector

        return lambda: HardenedDetector(FALCON9)
    if name != "base":
        raise ValueError(f"unknown detector {name!r}")
    l1, l2, alert = L1Config.load(FALCON9), L2Model.load(FALCON9), AlertConfig.load(FALCON9.name)
    return lambda: BaseDetector(FALCON9, l1_config=l1, l2=l2 or False, alert=alert)


class Worker:
    def __init__(self, config: WorkerConfig, members: Callable[[], set[int]],
                 make_detector: Callable[[], Detector]):
        self.config = config
        self.members = members
        self.make_detector = make_detector
        slugs = holdout_slugs() if config.launches == "holdout" else all_slugs()
        self.specs = fleet_specs(config.fleet_size, slugs, config.attack_fraction)
        self.streams: dict[str, VehicleStream] = {}
        self.shard_members: frozenset[int] = frozenset()
        self.ready = False

        self.m_vehicles = Gauge("fleet_vehicles", "Vehicles this worker owns")
        self.m_members = Gauge("fleet_members", "Live workers this worker last sharded over")
        self.m_target_fps = Gauge("fleet_target_frames_per_second", "Frames per second needed to keep up")
        self.m_lag = Gauge("fleet_lag_seconds", "How far behind real time this worker is")
        self.m_frames = Counter("fleet_frames_total", "Frames scored")
        self.m_alert_frames = Counter("fleet_alert_frames_total", "Frames with an active alert")
        self.m_moved = Counter("fleet_vehicles_moved_total", "Vehicles gained or dropped by resharding")
        self.m_member_errors = Counter("fleet_member_lookup_errors_total", "Failed live-worker lookups")
        self.m_flights_attacked = Counter("fleet_flights_attacked_total", "Finished flights under attack")
        self.m_flights_detected = Counter("fleet_flights_detected_total", "Attacked flights alerted after onset")
        self.m_flights_clean = Counter("fleet_flights_clean_total", "Finished clean flights")
        self.m_false_alerts = Counter("fleet_flights_false_alert_total",
                                      "Flights with an alert before any attack began")
        self.m_frame_latency = Histogram("fleet_frame_seconds", "Time to score one frame (link + detector)",
                                         FRAME_LATENCY_BUCKETS)
        self.m_detection_latency = Histogram("fleet_detection_latency_seconds",
                                             "Mission time from attack onset to first alert",
                                             DETECTION_LATENCY_BUCKETS)

    def metrics_text(self) -> str:
        return render([
            self.m_vehicles, self.m_members, self.m_target_fps, self.m_lag, self.m_frames, self.m_alert_frames,
            self.m_moved, self.m_member_errors, self.m_flights_attacked, self.m_flights_detected,
            self.m_flights_clean, self.m_false_alerts, self.m_frame_latency, self.m_detection_latency,
        ])

    # ------------------------------------------------------------------ sharding

    def reshard(self, members: frozenset[int]) -> None:
        mine = {s.vehicle_id: s for s in self.specs if owner(s.vehicle_id, members) == self.config.ordinal}
        dropped = [v for v in self.streams if v not in mine]
        for vehicle_id in dropped:
            del self.streams[vehicle_id]
        added = [v for v in mine if v not in self.streams]
        for vehicle_id in added:
            self.streams[vehicle_id] = VehicleStream(mine[vehicle_id], self.make_detector)
        if self.ready and (added or dropped):
            log.info("resharded over workers %s -> %s: +%d / -%d vehicles", sorted(self.shard_members),
                     sorted(members), len(added), len(dropped))
        self.m_moved.inc(len(added) + len(dropped) if self.ready else 0)
        self.shard_members = members
        self.m_members.set(len(members))
        self.m_vehicles.set(len(self.streams))
        self.m_target_fps.set(len(self.streams) * RATE_HZ * self.config.speed)
        self.ready = True

    def maybe_reshard(self) -> None:
        try:
            live = self.members()
        except Exception as exc:   # API blip: keep the current shard map rather than dropping vehicles
            self.m_member_errors.inc()
            log.warning("worker lookup failed, keeping workers %s: %s", sorted(self.shard_members), exc)
            return
        # Always count ourselves: a new pod only turns Ready after taking its share, so it can't wait to see
        # itself in the list. Until the others see it Ready too, both score its vehicles: a short overlap
        # instead of a gap.
        members = frozenset(live) | {self.config.ordinal}
        if members != self.shard_members:
            self.reshard(members)

    # ------------------------------------------------------------------ scoring

    def tick(self) -> None:
        """One frame for every owned vehicle."""
        for stream in self.streams.values():
            start = time.perf_counter()
            verdict, result = stream.step()
            self.m_frame_latency.observe(time.perf_counter() - start)
            self.m_frames.inc()
            if verdict.alert:
                self.m_alert_frames.inc()
            if result is not None:
                self._record(result)

    def _record(self, result: FlightResult) -> None:
        if result.attacked:
            self.m_flights_attacked.inc()
            if result.detected:
                self.m_flights_detected.inc()
                self.m_detection_latency.observe(result.latency_s)
        else:
            self.m_flights_clean.inc()
        if result.false_alert:
            self.m_false_alerts.inc()

    def run(self, stop: threading.Event, max_ticks: int | None = None) -> None:
        period = 1.0 / (RATE_HZ * self.config.speed)
        next_tick = time.monotonic()
        last_reshard = float("-inf")
        ticks = 0
        while not stop.is_set() and (max_ticks is None or ticks < max_ticks):
            if time.monotonic() - last_reshard >= self.config.reshard_every_s:
                self.maybe_reshard()
                last_reshard = time.monotonic()
            self.tick()
            ticks += 1
            next_tick += period
            lag = time.monotonic() - next_tick
            self.m_lag.set(max(lag, 0.0))
            if lag < 0:
                stop.wait(-lag)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = WorkerConfig.from_env()
    if config.worker_selector and k8s.in_cluster():
        def members() -> set[int]:
            return k8s.live_workers(config.worker_selector)
    else:
        def members() -> set[int]:
            return set(range(config.replicas))

    worker = Worker(config, members, detector_factory(config.detector))
    serve(config.metrics_port, worker.metrics_text, lambda: worker.ready)

    # As PID 1 in a container, Python gets no default SIGTERM handling: without this, `kubectl scale` down would
    # wait out the whole termination grace period and then SIGKILL the pod.
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log.info("worker %d up: fleet %d, speed %.1fx, detector %s", config.ordinal, config.fleet_size,
             config.speed, config.detector)
    worker.run(stop)
    log.info("worker %d stopping", config.ordinal)


if __name__ == "__main__":
    main()

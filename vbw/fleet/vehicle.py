"""One vehicle in the fleet: the same path a single run takes in the console, minus the UI.

    Falcon 9 replay (seeded sensor noise) -> MITM attack (or none) -> CCSDS-style frame with a valid CRC
    -> LinkMonitor (L0) -> detector

Each vehicle flies its launch, then flies it again with a new noise seed, forever. A flight is scored when it ends:
an attacked flight counts as detected if an alert fired at or after the attack began; any alert on a clean
flight, or before the attack began, is a false alert.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Iterator

from vbw.attacker.registry import make_attack
from vbw.link.codec import LinkMonitor, encode
from vbw.schema import FALCON9, Attack, Detector, Frame, Verdict
from vbw.sources.falcon9 import Falcon9Source

# Scripted attacks only: the PPO attackers need torch + SB3 (~1 GB per process), too heavy to run per worker.
FLEET_ATTACKS = ("bias", "drift", "replay", "freeze", "coordinated")
ATTACK_ONSET_FRACTION = 0.35   # same protocol as results/detector_eval*.json: above the atmosphere


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_id: str
    slug: str                  # Falcon 9 launch it replays
    seed: int
    attack: str | None         # scripted attack name, or None for a clean vehicle


def fleet_specs(size: int, slugs: list[str], attack_fraction: float) -> list[VehicleSpec]:
    """The whole fleet, identical in every worker: vehicle i flies slugs[i % len(slugs)] with seed i.

    Which vehicles are attacked is a hash of the id, so it doesn't line up with the launch rotation.
    """
    specs = []
    for i in range(size):
        vehicle_id = f"f9-{i:05d}"
        attacked = zlib.crc32(f"attack/{vehicle_id}".encode()) % 10_000 < attack_fraction * 10_000
        attack = FLEET_ATTACKS[i % len(FLEET_ATTACKS)] if attacked else None
        specs.append(VehicleSpec(vehicle_id, slugs[i % len(slugs)], seed=i, attack=attack))
    return specs


@dataclass
class FlightResult:
    attacked: bool
    detected: bool             # attacked flight: alert at or after onset
    false_alert: bool          # alert on a clean flight, or before onset
    latency_s: float | None    # onset -> first alert, for detected flights


class VehicleStream:
    def __init__(self, spec: VehicleSpec, make_detector: Callable[[], Detector]):
        self.spec = spec
        self.detector = make_detector()
        self.flights = 0
        self._start_flight()

    def _start_flight(self) -> None:
        # A new seed per flight, so repeated flights of the same launch see different sensor noise.
        source = Falcon9Source(self.spec.slug, seed=self.spec.seed * 1000 + self.flights)
        self.onset_t: float | None = None
        self.attack: Attack | None = None
        if self.spec.attack:
            self.onset_t = ATTACK_ONSET_FRACTION * source.duration_s
            self.attack = make_attack(self.spec.attack, FALCON9, start_t=self.onset_t, intensity=1.0)
            self.attack.reset(seed=self.spec.seed)
        self._frames: Iterator[Frame] = source.frames()
        self.link = LinkMonitor(self.spec.vehicle_id)
        self.detector.reset()
        self.first_alert_t: float | None = None
        self.early_alert = False

    def step(self) -> tuple[Verdict, FlightResult | None]:
        """Process one frame. Returns the verdict, plus the flight's result when this frame ended a flight."""
        try:
            frame = next(self._frames)
        except StopIteration:
            result = self.result()
            self.flights += 1
            self._start_flight()
            return self.step()[0], result

        frame.vehicle_id = self.spec.vehicle_id
        reported = self.attack(frame) if self.attack else frame
        received, l0 = self.link.receive(encode(reported))   # the MITM re-encoded it, so the CRC is valid
        verdict = self.detector.update(received, l0)
        if verdict.alert:
            if self.onset_t is None or received.t < self.onset_t:
                self.early_alert = True
            elif self.first_alert_t is None:
                self.first_alert_t = received.t
        return verdict, None

    def result(self) -> FlightResult:
        attacked = self.onset_t is not None
        detected = attacked and self.first_alert_t is not None
        latency = self.first_alert_t - self.onset_t if detected else None
        return FlightResult(attacked, detected, self.early_alert, latency)

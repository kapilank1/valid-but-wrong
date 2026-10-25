"""Replay telemetry scenarios as UDP frames to the C++ edge monitor (edge/SPEC.md section 6).

    uv run python scripts/udp_replay.py --scenario falcon9:ses-9 --attack bias --attack-start 190 --port 14650
    uv run python scripts/udp_replay.py --vehicles 8 --attacked 4 --attack bias --rate-hz 50 --truth truth.json
    uv run python scripts/udp_replay.py --speed 0            # as fast as the socket takes them

Each vehicle's stream is built exactly like the pipeline's link (vbw/pipeline/session.py): the source frame is
encoded and decoded (float32 wire rounding), the MITM attack rewrites the decoded frame, and the result is
re-encoded with a fresh, valid CRC. Everything is precomputed before the first packet goes out, so a slow attack
(PPO runs a policy plus a shadow detector per frame) can't stretch the send timing.

Pacing: frame k of a vehicle is due at (t_k - t_0) / speed, or at k / rate_hz / speed with --rate-hz (e.g. 50 to
replay a 10 Hz flight as a 50 Hz stream; mission time t in the frames is unchanged). --speed 0 sends with no
pacing at all. Vehicles are interleaved frame by frame. --vehicles N makes N copies of each scenario with distinct
vehicle_ids (<id>-<i>) and noise seeds (seed + i), so every copy is a different but equally real flight.

--truth writes the ground truth the evaluation joins against: per vehicle the vehicle_id, its vid_hash (the
CRC-16 the monitor keys vehicles by), the attack and its onset, and how many frames were sent.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vbw.attacker.registry import make_attack
from vbw.link.codec import decode, encode, vehicle_hash
from vbw.sources.registry import make_source

DEFAULT_SCENARIO = "falcon9:ses-9"
DEFAULT_PORT = 14650
DEFAULT_ATTACK_START = 190.0   # ses-9: just after max-Q, the hero demo's attack time


@dataclass
class Vehicle:
    """One precomputed stream: the exact UDP payloads to send, plus what the evaluation needs to know about them."""

    vehicle_id: str
    scenario: str
    schema: str
    schema_id: int
    seed: int
    rate_hz: float
    attack: str = "none"
    attack_params: dict[str, Any] = field(default_factory=dict)
    packets: list[bytes] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    onset_t: float | None = None       # first frame whose bytes differ from what the vehicle sent

    @property
    def vid_hash(self) -> int:
        return vehicle_hash(self.vehicle_id)

    @property
    def attack_start(self) -> float | None:
        return None if self.attack == "none" else float(self.attack_params.get("start_t", 0.0))

    def truth(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "vid_hash": f"0x{self.vid_hash:04x}",
            "scenario": self.scenario,
            "schema": self.schema,
            "schema_id": self.schema_id,
            "seed": self.seed,
            "attack": self.attack,
            "attack_params": dict(self.attack_params),
            "attack_start": self.attack_start,
            "onset_t": self.onset_t,
            "frames": len(self.packets),
            "first_t": self.t[0] if self.t else None,
            "last_t": self.t[-1] if self.t else None,
        }


def build_vehicle(scenario: str = DEFAULT_SCENARIO, seed: int = 0, vehicle_id: str | None = None,
                  attack: str = "none", attack_start: float = DEFAULT_ATTACK_START, intensity: float = 1.0,
                  max_frames: int | None = None) -> Vehicle:
    """Precompute one vehicle's packets: source -> encode/decode -> attack.apply -> encode (valid CRC)."""
    source = make_source(scenario, seed=seed)
    schema = source.schema
    vid = vehicle_id or source.vehicle_id
    params = {} if attack == "none" else {"start_t": float(attack_start), "intensity": float(intensity)}
    if attack.startswith("ppo"):
        import torch

        torch.set_num_threads(1)   # deterministic, and the policy is a tiny MLP
    mitm = make_attack(attack, schema, **params)
    if getattr(mitm, "loaded", True) is False:
        raise SystemExit(f"attack {attack!r}: no trained policy on disk for {schema.name}")
    mitm.reset(seed)

    vehicle = Vehicle(vid, scenario, schema.name, schema.schema_id, seed, float(source.rate_hz), attack, params)
    for i, frame in enumerate(source.frames()):
        if max_frames is not None and i >= max_frames:
            break
        frame.vehicle_id = vid
        downlink = encode(frame)
        truth, _ = decode(downlink, vid)
        packet = encode(mitm.apply(truth.copy()))
        if vehicle.onset_t is None and packet != downlink:
            vehicle.onset_t = truth.t
        vehicle.packets.append(packet)
        vehicle.t.append(truth.t)
    return vehicle


def fleet_ids(scenarios: list[str], copies: int) -> list[tuple[str, str | None]]:
    """(scenario, vehicle_id or None for the source's own id) for every vehicle, ids unique."""
    out, used = [], set()
    for j, scenario in enumerate(scenarios):
        base = make_source(scenario, seed=0).vehicle_id if (copies > 1 or scenarios.count(scenario) > 1) else None
        for i in range(copies):
            if base is None:
                out.append((scenario, None))
                continue
            vid = f"{base}-{i}" if copies > 1 else base
            if vid in used:
                vid = f"{vid}-{j}"
            used.add(vid)
            out.append((scenario, vid))
    return out


def build_fleet(scenarios: list[str], copies: int = 1, seed: int = 0, attack: str = "none",
                attack_start: float = DEFAULT_ATTACK_START, intensity: float = 1.0, attacked: int | None = None,
                max_frames: int | None = None, log: bool = False) -> list[Vehicle]:
    """Every vehicle of a replay. Vehicle i gets noise seed `seed + i`; only the first `attacked` get the attack."""
    vehicles = []
    for i, (scenario, vid) in enumerate(fleet_ids(scenarios, copies)):
        name = attack if attacked is None or i < attacked else "none"
        t0 = time.perf_counter()
        v = build_vehicle(scenario, seed + i, vid, name, attack_start, intensity, max_frames)
        if log:
            print(f"  {v.vehicle_id} (vid 0x{v.vid_hash:04x}, seed {v.seed}, {name}): "
                  f"{len(v.packets)} frames in {time.perf_counter() - t0:.1f}s")
        vehicles.append(v)
    hashes = [v.vid_hash for v in vehicles]
    if len(set(hashes)) != len(hashes):
        raise SystemExit(f"vid_hash collision among {[v.vehicle_id for v in vehicles]}; pick other ids")
    return vehicles


def schedule(vehicles: list[Vehicle], rate_hz: float | None = None) -> list[tuple[float, int, int]]:
    """(offset s at speed 1, vehicle index, frame index), in send order: by due time, then vehicle (interleaved)."""
    order = []
    for v, veh in enumerate(vehicles):
        t0 = veh.t[0] if veh.t else 0.0
        for k, t in enumerate(veh.t):
            order.append((k / rate_hz if rate_hz else t - t0, v, k))
    order.sort(key=lambda item: (item[0], item[1], item[2]))
    return order


def _wait_until(due: float) -> None:
    """Sleep most of the way, then spin: time.sleep alone overshoots by up to a millisecond on Windows."""
    while True:
        left = due - time.perf_counter()
        if left <= 0:
            return
        if left > 0.002:
            time.sleep(left - 0.0015)


def send(vehicles: list[Vehicle], host: str = "127.0.0.1", port: int = DEFAULT_PORT, speed: float = 1.0,
         rate_hz: float | None = None) -> dict[str, Any]:
    """Send every vehicle's packets; returns a summary (frames, elapsed, achieved and target rate)."""
    order = schedule(vehicles, rate_hz)
    addr = (host, port)
    sent, errors = 0, 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        start = time.perf_counter()
        for offset, v, k in order:
            if speed > 0:
                _wait_until(start + offset / speed)
            try:
                sock.sendto(vehicles[v].packets[k], addr)
                sent += 1
            except OSError:
                errors += 1
        elapsed = time.perf_counter() - start
    span = order[-1][0] if order else 0.0
    return {
        "frames_sent": sent,
        "send_errors": errors,
        "vehicles": len(vehicles),
        "elapsed_s": elapsed,
        "achieved_frames_per_s": sent / elapsed if elapsed > 0 else None,
        "target_frames_per_s": (len(order) / (span / speed)) if speed > 0 and span > 0 else None,
        "speed": speed,
        "rate_hz": rate_hz,
    }


def truth_doc(vehicles: list[Vehicle], stats: dict[str, Any] | None = None, host: str = "127.0.0.1",
              port: int = DEFAULT_PORT) -> dict[str, Any]:
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host,
        "port": port,
        "send": stats or {},
        "vehicles": [v.truth() for v in vehicles],
    }


def write_truth(path: Path, vehicles: list[Vehicle], stats: dict[str, Any] | None = None,
                host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(truth_doc(vehicles, stats, host, port), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", default=None,
                    help=f"scenario id, repeatable (default {DEFAULT_SCENARIO})")
    ap.add_argument("--vehicles", type=int, default=1, help="copies of each scenario (distinct ids and seeds)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed of the first vehicle; vehicle i uses seed + i")
    ap.add_argument("--attack", default="none", help="attack name from vbw.attacker.registry (none, bias, ppo, ...)")
    ap.add_argument("--attack-start", type=float, default=DEFAULT_ATTACK_START, help="attack start, mission s")
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--attacked", type=int, default=None, help="only the first K vehicles are attacked (default all)")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 2 = twice as fast, 0 = no pacing")
    ap.add_argument("--rate-hz", type=float, default=None, help="send at this per-vehicle frame rate instead of frame t")
    ap.add_argument("--max-frames", type=int, default=None, help="only the first N frames of each vehicle")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--truth", type=Path, default=None, help="write the ground-truth JSON here")
    args = ap.parse_args()
    if args.speed < 0 or (args.rate_hz is not None and args.rate_hz <= 0):
        ap.error("--speed must be >= 0 and --rate-hz > 0")

    scenarios = args.scenario or [DEFAULT_SCENARIO]
    print(f"precomputing {len(scenarios) * args.vehicles} vehicle(s)...")
    vehicles = build_fleet(scenarios, args.vehicles, args.seed, args.attack, args.attack_start, args.intensity,
                           args.attacked, args.max_frames, log=True)
    total = sum(len(v.packets) for v in vehicles)
    pace = "max rate" if args.speed == 0 else (f"{args.rate_hz:g} Hz per vehicle" if args.rate_hz else "frame t")
    print(f"sending {total} frames to {args.host}:{args.port} ({pace}, speed {args.speed:g})")
    stats = send(vehicles, args.host, args.port, args.speed, args.rate_hz)
    rate = stats["achieved_frames_per_s"]
    print(f"sent {stats['frames_sent']} frames from {stats['vehicles']} vehicle(s) in {stats['elapsed_s']:.2f}s "
          f"({rate:.0f} frames/s)" + (f", {stats['send_errors']} send errors" if stats["send_errors"] else ""))
    if args.truth:
        write_truth(args.truth, vehicles, stats, args.host, args.port)
        print(f"truth -> {args.truth}")


if __name__ == "__main__":
    main()

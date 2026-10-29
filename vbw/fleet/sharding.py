"""Which worker owns which vehicle.

A vehicle's detector keeps ~61 s of history (the longest L1 window is 610 frames), so every frame of one vehicle
must reach the same worker. Workers are StatefulSet pods with stable ordinals (worker-0, worker-1, ...); each one
computes the same ownership function over the same member set and keeps only its own vehicles, so no coordinator
or queue is needed.

Members are the ordinals of the workers that are actually up (Ready pods), not the desired replica count: if a
pod is Pending or crash-looping, its vehicles go to the survivors instead of going unmonitored.

Ownership uses rendezvous (highest-random-weight) hashing rather than `hash % n`: a vehicle belongs to the member
with the highest hash(vehicle, member). Adding a member moves only the vehicles it wins (~1/(n+1) of them); losing
one moves only that member's vehicles, spread over the rest, and they move back when it returns. Modulo would move
~n/(n+1) of all vehicles on every change. Every moved vehicle restarts its detector history, so fewer moves means
less blind time.
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def _weight(vehicle_id: str, shard: int) -> int:
    # Not Python's hash(): hash() of a str is salted per process, so two pods would disagree. Not crc32 either:
    # CRC is linear, so crc(id/0) and crc(id/1) differ by a constant XOR and the "highest weight" pick would
    # depend on a few bits only (lopsided shards). Rendezvous hashing needs a well-mixed hash.
    return int.from_bytes(hashlib.blake2b(f"{vehicle_id}/{shard}".encode(), digest_size=8).digest(), "big")


def owner(vehicle_id: str, members: Iterable[int]) -> int:
    """The member ordinal that owns vehicle_id. Every process given the same members computes the same answer."""
    members = list(members)
    if not members:
        raise ValueError("need at least one member")
    return max(members, key=lambda shard: _weight(vehicle_id, shard))


def owned_by(vehicle_ids: list[str], ordinal: int, members: Iterable[int]) -> list[str]:
    members = list(members)
    return [v for v in vehicle_ids if owner(v, members) == ordinal]


def ordinal_from_hostname(hostname: str) -> int:
    """StatefulSet pods are named <statefulset>-<ordinal>, and the pod name is the container's hostname."""
    name, sep, ordinal = hostname.rpartition("-")
    if not sep or not name or not ordinal.isdigit():
        raise ValueError(f"hostname {hostname!r} is not a StatefulSet pod name like 'worker-3'")
    return int(ordinal)

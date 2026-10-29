import zlib

import pytest

from vbw.fleet.sharding import ordinal_from_hostname, owned_by, owner

FLEET = [f"f9-{i:05d}" for i in range(2000)]


def test_every_vehicle_has_exactly_one_owner():
    members = range(4)
    owned = [owned_by(FLEET, ordinal, members) for ordinal in members]
    assert sorted(v for part in owned for v in part) == sorted(FLEET)


def test_shards_are_roughly_balanced():
    counts = [len(owned_by(FLEET, ordinal, range(4))) for ordinal in range(4)]
    assert all(400 <= c <= 600 for c in counts), counts


def test_owner_is_deterministic_across_processes():
    # blake2b-based, so the answer is the same in every pod (Python's salted hash() would differ) and
    # doesn't depend on the order members are listed in.
    assert owner("f9-00042", range(8)) == owner("f9-00042", [7, 6, 5, 4, 3, 2, 1, 0])
    assert owner("f9-00042", [3]) == 3


def test_scale_up_moves_only_the_new_shards_share():
    before = {v: owner(v, range(4)) for v in FLEET}
    after = {v: owner(v, range(5)) for v in FLEET}
    moved = [v for v in FLEET if before[v] != after[v]]
    assert all(after[v] == 4 for v in moved)          # vehicles only move to the new worker
    assert 0.15 < len(moved) / len(FLEET) < 0.25      # ~1/5, where hash % n would move ~4/5

    modulo_moved = sum(zlib.crc32(v.encode()) % 4 != zlib.crc32(v.encode()) % 5 for v in FLEET)
    assert modulo_moved > 3 * len(moved)


def test_losing_a_worker_moves_only_its_vehicles_and_they_come_back():
    full = {v: owner(v, range(4)) for v in FLEET}
    without_2 = {v: owner(v, [0, 1, 3]) for v in FLEET}
    moved = [v for v in FLEET if full[v] != without_2[v]]
    assert moved and all(full[v] == 2 for v in moved)          # only worker-2's vehicles moved
    assert {without_2[v] for v in moved} == {0, 1, 3}           # spread over every survivor
    assert {v: owner(v, [0, 1, 2, 3]) for v in FLEET} == full   # worker-2 back: same map as before


def test_owner_rejects_no_members():
    with pytest.raises(ValueError):
        owner("f9-00000", [])


@pytest.mark.parametrize("hostname, ordinal", [("worker-0", 0), ("worker-12", 12), ("fleet-worker-3", 3)])
def test_ordinal_from_statefulset_hostname(hostname, ordinal):
    assert ordinal_from_hostname(hostname) == ordinal


@pytest.mark.parametrize("hostname", ["worker", "worker-", "-3", "worker-abc", "Rajeevs-MacBook"])
def test_ordinal_rejects_non_statefulset_hostnames(hostname):
    with pytest.raises(ValueError):
        ordinal_from_hostname(hostname)

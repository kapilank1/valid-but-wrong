"""CCSDS-inspired binary telemetry frame with CRC-16/CCITT-FALSE.

Layout (big-endian):
    sync      4B  0x1ACFFC1D (CCSDS attached sync marker)
    version   1B
    schema_id 1B
    vid_hash  2B  CRC-16 of the vehicle_id string (APID stand-in)
    seq       2B  uint16, wraps
    t         8B  float64 mission time
    n         1B  channel count
    values    4B*n float32 in schema channel order
    crc       2B  CRC-16/CCITT-FALSE over everything before it

The CRC proves the bytes weren't corrupted in transit. It says nothing about whether they are true.
"""

from __future__ import annotations

import struct

from vbw.schema import SCHEMAS, Frame, L0Result

SYNC = 0x1ACFFC1D
VERSION = 1
_HEADER = struct.Struct(">IBBHHdB")


class DecodeError(ValueError):
    pass


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), as used by CCSDS TM frames."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def vehicle_hash(vehicle_id: str) -> int:
    return crc16_ccitt(vehicle_id.encode())


def encode(frame: Frame) -> bytes:
    schema = SCHEMAS[frame.schema_id]
    values = [float(frame.channels[c]) for c in schema.channels]
    body = _HEADER.pack(
        SYNC, VERSION, frame.schema_id, vehicle_hash(frame.vehicle_id),
        frame.seq & 0xFFFF, frame.t, len(values),
    ) + struct.pack(f">{len(values)}f", *values)
    return body + struct.pack(">H", crc16_ccitt(body))


def decode(packet: bytes, vehicle_id: str) -> tuple[Frame, L0Result]:
    """Decode a packet. Raises DecodeError only if it is structurally unparseable;
    integrity failures (bad sync/CRC) are reported in L0Result instead."""
    if len(packet) < _HEADER.size + 2:
        raise DecodeError(f"packet too short: {len(packet)} bytes")
    sync, version, schema_id, _vid, seq, t, n = _HEADER.unpack_from(packet)
    schema = SCHEMAS.get(schema_id)
    if schema is None or n != len(schema.channels):
        raise DecodeError(f"unknown schema {schema_id} with {n} channels")
    expected_len = _HEADER.size + 4 * n + 2
    if len(packet) != expected_len:
        raise DecodeError(f"length {len(packet)} != {expected_len}")

    values = struct.unpack_from(f">{n}f", packet, _HEADER.size)
    (crc,) = struct.unpack_from(">H", packet, expected_len - 2)

    l0 = L0Result()
    if sync != SYNC:
        l0.sync_ok = False
        l0.reasons.append(f"bad sync 0x{sync:08X}")
    if crc != crc16_ccitt(packet[:-2]):
        l0.crc_ok = False
        l0.reasons.append("CRC mismatch")

    frame = Frame(vehicle_id, schema_id, seq, t, dict(zip(schema.channels, map(float, values))))
    return frame, l0


class LinkMonitor:
    """Ground-side L0 checks that need state: sequence continuity and monotonic time."""

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.reset()

    def reset(self) -> None:
        self._last_seq: int | None = None
        self._last_t: float | None = None

    def receive(self, packet: bytes) -> tuple[Frame, L0Result]:
        frame, l0 = decode(packet, self.vehicle_id)
        if self._last_seq is not None and frame.seq != (self._last_seq + 1) & 0xFFFF:
            l0.seq_ok = False
            l0.reasons.append(f"seq jump {self._last_seq}->{frame.seq}")
        if self._last_t is not None and frame.t <= self._last_t:
            l0.time_ok = False
            l0.reasons.append(f"time not monotonic {self._last_t:.3f}->{frame.t:.3f}")
        self._last_seq, self._last_t = frame.seq, frame.t
        return frame, l0

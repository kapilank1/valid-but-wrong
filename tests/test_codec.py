import pytest

from vbw.link.codec import DecodeError, LinkMonitor, crc16_ccitt, decode, encode
from vbw.schema import FALCON9, Frame


def make_frame(seq=0, t=0.0, speed=100.0):
    channels = {c: 0.0 for c in FALCON9.channels}
    channels["speed"] = speed
    return Frame("f9-test", FALCON9.schema_id, seq, t, channels)


def test_crc16_ccitt_false_check_value():
    # Standard check value for CRC-16/CCITT-FALSE over "123456789"
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_roundtrip_preserves_values():
    frame = make_frame(seq=7, t=12.5, speed=1234.5)
    decoded, l0 = decode(encode(frame), "f9-test")
    assert l0.ok
    assert decoded.seq == 7 and decoded.t == 12.5
    assert decoded.channels["speed"] == pytest.approx(1234.5)


def test_corrupted_byte_fails_crc():
    packet = bytearray(encode(make_frame()))
    packet[25] ^= 0xFF
    _, l0 = decode(bytes(packet), "f9-test")
    assert not l0.crc_ok and not l0.ok


def test_forged_frame_reencoded_passes_l0():
    """The core premise: a MITM that rewrites values and recomputes the CRC is invisible to L0."""
    monitor = LinkMonitor("f9-test")
    monitor.receive(encode(make_frame(seq=0, t=0.0)))
    frame, _ = decode(encode(make_frame(seq=1, t=0.1)), "f9-test")
    frame.channels["speed"] += 500.0  # a lie
    _, l0 = monitor.receive(encode(frame))
    assert l0.ok


def test_seq_gap_and_time_regression_flagged():
    monitor = LinkMonitor("f9-test")
    monitor.receive(encode(make_frame(seq=0, t=1.0)))
    _, l0 = monitor.receive(encode(make_frame(seq=5, t=0.5)))
    assert not l0.seq_ok and not l0.time_ok


def test_seq_wraps():
    monitor = LinkMonitor("f9-test")
    monitor.receive(encode(make_frame(seq=65535, t=0.0)))
    _, l0 = monitor.receive(encode(make_frame(seq=0, t=0.1)))
    assert l0.seq_ok


def test_truncated_packet_raises():
    with pytest.raises(DecodeError):
        decode(encode(make_frame())[:10], "f9-test")

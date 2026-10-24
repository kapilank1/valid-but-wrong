import socket
import threading
import time

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mav

from vbw.schema import QUAD
from vbw.sources.mavlink_bridge import (
    MG_TO_MS2,
    BaroReference,
    BridgeState,
    SitlSource,
    message_to_channels,
    pressure_altitude,
)

ENCODER = mav.MAVLink(None, srcSystem=1, srcComponent=1)


def over_the_wire(msg):
    """Pack and re-parse, so the test sees exactly what a receiver would (float32 fields, wire units)."""
    parser = mav.MAVLink(None)
    return parser.parse_buffer(msg.pack(ENCODER))[0]


def local_position(x=1.0, y=2.0, z=-10.0, vx=0.5, vy=-0.25, vz=-1.5):
    return over_the_wire(mav.MAVLink_local_position_ned_message(1000, x, y, z, vx, vy, vz))


def attitude():
    return over_the_wire(mav.MAVLink_attitude_message(1000, 0.1, -0.2, 2.5, 0.01, 0.02, -0.03))


def raw_imu(zacc=-1000):
    return over_the_wire(mav.MAVLink_raw_imu_message(1000, 50, -20, zacc, 0, 0, 0, 0, 0, 0))


def scaled_pressure(press_hpa=1013.25, temp_cdeg=1500):
    return over_the_wire(mav.MAVLink_scaled_pressure_message(1000, press_hpa, 0.0, temp_cdeg))


def sys_status(mv=15800):
    return over_the_wire(mav.MAVLink_sys_status_message(0, 0, 0, 0, mv, -1, -1, 0, 0, 0, 0, 0, 0))


def vfr_hud(groundspeed=3.0):
    return over_the_wire(mav.MAVLink_vfr_hud_message(0.0, groundspeed, 90, 50, 10.0, 1.5))


def all_messages():
    return [local_position(), attitude(), raw_imu(), scaled_pressure(), sys_status()]


def test_local_position_ned_flips_down_to_up():
    ch = message_to_channels(local_position())
    assert ch == pytest.approx({"pn": 1.0, "pe": 2.0, "alt": 10.0, "vn": 0.5, "ve": -0.25, "vu": 1.5})


def test_attitude_maps_angles_and_body_rates():
    ch = message_to_channels(attitude())
    assert ch == pytest.approx({"roll": 0.1, "pitch": -0.2, "yaw": 2.5, "p": 0.01, "q": 0.02, "r": -0.03})


@pytest.mark.parametrize("kind", ["RAW_IMU", "SCALED_IMU"])
def test_imu_milli_g_to_specific_force(kind):
    if kind == "RAW_IMU":
        msg = raw_imu()
    else:
        msg = over_the_wire(mav.MAVLink_scaled_imu_message(1000, 50, -20, -1000, 0, 0, 0, 0, 0, 0))
    ch = message_to_channels(msg)
    assert ch["az"] == pytest.approx(-9.80665)  # level and at rest: QUAD convention az ~ -g
    assert ch["ax"] == pytest.approx(50 * MG_TO_MS2) and ch["ay"] == pytest.approx(-20 * MG_TO_MS2)


def test_second_imu_and_baro_are_ignored():
    imu2 = over_the_wire(mav.MAVLink_scaled_imu2_message(1000, 0, 0, -1000, 0, 0, 0, 0, 0, 0))
    baro2 = over_the_wire(mav.MAVLink_scaled_pressure2_message(1000, 1000.0, 0.0, 1500))
    assert message_to_channels(imu2) == {} and message_to_channels(baro2) == {}


def test_pressure_altitude_matches_standard_atmosphere():
    ref = BaroReference(press_hpa=1013.25, temp_k=288.15)
    assert pressure_altitude(1013.25, ref) == pytest.approx(0.0, abs=1e-9)
    assert pressure_altitude(1001.29, ref) == pytest.approx(100.0, abs=0.5)  # ISA: 100 m -> 1001.29 hPa
    msg = scaled_pressure(press_hpa=1001.29)
    assert message_to_channels(msg, ref)["baro_alt"] == pytest.approx(100.0, abs=0.5)


def test_battery_and_groundspeed():
    assert message_to_channels(sys_status(15800)) == {"battery_v": pytest.approx(15.8)}
    assert message_to_channels(sys_status(0xFFFF)) == {}  # "unknown" sentinel
    assert message_to_channels(vfr_hud(3.0)) == {"groundspeed": pytest.approx(3.0)}


def test_unrelated_messages_map_to_nothing():
    heartbeat = over_the_wire(mav.MAVLink_heartbeat_message(2, 3, 0, 0, 0, 3))
    assert message_to_channels(heartbeat) == {}


def test_bridge_state_waits_for_every_channel_then_snapshots_in_schema_order():
    state = BridgeState()
    messages = all_messages()
    for msg in messages[:-1]:
        state.update(msg)
        assert not state.ready()
    state.update(messages[-1])
    assert state.ready()
    snap = state.snapshot()
    assert tuple(snap) == QUAD.channels
    assert snap["groundspeed"] == pytest.approx(0.559, abs=1e-3)  # hypot(0.5, -0.25) until VFR_HUD arrives
    assert snap["baro_alt"] == pytest.approx(0.0)                  # first pressure reading is the reference
    state.update(vfr_hud(3.0))
    assert state.snapshot()["groundspeed"] == pytest.approx(3.0)
    state.update(scaled_pressure(press_hpa=1001.29))
    assert state.snapshot()["baro_alt"] > 90.0


def test_constructing_sitl_source_does_not_listen():
    source = SitlSource(udp_port=14550)
    assert source.vehicle_id == "sitl-udp-14550" and source.schema is QUAD and source.rate_hz == 20.0


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_loopback_stream_produces_quad_frames():
    """End to end over a real UDP socket with a fake autopilot (no SITL needed)."""
    from pymavlink import mavutil

    port = free_udp_port()
    stop = threading.Event()

    def autopilot():
        out = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1)
        while not stop.is_set():
            out.mav.local_position_ned_send(0, 1.0, 2.0, -10.0, 0.5, -0.25, -1.5)
            out.mav.attitude_send(0, 0.1, -0.2, 2.5, 0.01, 0.02, -0.03)
            out.mav.raw_imu_send(0, 0, 0, -1000, 0, 0, 0, 0, 0, 0)
            out.mav.scaled_pressure_send(0, 1013.25, 0.0, 1500)
            out.mav.sys_status_send(0, 0, 0, 0, 15800, -1, -1, 0, 0, 0, 0, 0, 0)
            time.sleep(0.01)
        out.close()

    sender = threading.Thread(target=autopilot, daemon=True)
    sender.start()
    try:
        source = SitlSource(udp_port=port, host="127.0.0.1", timeout_s=2.0, request_rates=False)
        frames = []
        for frame in source.frames():
            frames.append(frame)
            if len(frames) == 10:
                break
    finally:
        stop.set()
        sender.join(timeout=2.0)
    assert [f.seq for f in frames] == list(range(10))
    assert all(b.t > a.t for a, b in zip(frames, frames[1:]))
    assert frames[-1].channels["alt"] == pytest.approx(10.0) and frames[-1].channels["az"] == pytest.approx(-9.80665)
    assert tuple(frames[-1].channels) == QUAD.channels

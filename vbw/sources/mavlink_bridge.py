"""Live ArduPilot SITL (or any ArduPilot vehicle) over MAVLink -> QUAD frames at a fixed rate.

The bridge listens on UDP (MAVProxy / SITL `--out udp:<host>:14550`), keeps the latest value of every QUAD channel,
and emits a frame every 1 / rate_hz seconds (sample-and-hold resampling). Frame t is seconds since the first
emitted frame on the bridge's monotonic clock, so it is strictly increasing even if the autopilot clock jumps.

Message -> channel mapping (message_to_channels, a pure function):

    LOCAL_POSITION_NED   x, y, z, vx, vy, vz (NED)   -> pn, pe, alt = -z, vn, ve, vu = -vz
    ATTITUDE             roll, pitch, yaw, *speed     -> roll, pitch, yaw, p, q, r (body rates, rad/s)
    SCALED_IMU, RAW_IMU  xacc, yacc, zacc [mG]        -> ax, ay, az in m/s^2, body FRD specific force
                         (ArduPilot fills RAW_IMU with the same mG scaling as SCALED_IMU; level rest reads
                         zacc ~ -1000 mG -> az ~ -9.81, matching the QUAD convention. PX4 RAW_IMU is raw counts.)
    VFR_HUD              groundspeed [m/s]            -> groundspeed
    SCALED_PRESSURE      press_abs [hPa], temperature -> baro_alt, hypsometric altitude above the first reading
    SYS_STATUS           voltage_battery [mV]         -> battery_v (65535 = unknown, ignored)

Only IMU 0 and barometer 0 are used (SCALED_IMU2/3 and SCALED_PRESSURE2/3 are ignored) so channels never mix
sensors. Frames are emitted once every channel has been seen at least once; groundspeed falls back to
hypot(vn, ve) until a VFR_HUD arrives, which is exactly how ArduPilot computes it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from vbw.schema import QUAD, ChannelSchema, Frame
from vbw.sources.base import seq_at

log = logging.getLogger(__name__)

STANDARD_GRAVITY = 9.80665
MG_TO_MS2 = STANDARD_GRAVITY / 1000.0
UINT16_UNKNOWN = 0xFFFF
IMU_MESSAGES = ("SCALED_IMU", "RAW_IMU")
STREAMED_MESSAGES = ("LOCAL_POSITION_NED", "ATTITUDE", "SCALED_IMU", "RAW_IMU", "VFR_HUD", "SCALED_PRESSURE",
                     "SYS_STATUS")
# International Standard Atmosphere troposphere: lapse rate (K/m) and exponent R*L/(g*M).
ISA_LAPSE_K_PER_M = 0.0065
ISA_EXPONENT = 0.190263


@dataclass(frozen=True)
class BaroReference:
    """Ground pressure and temperature that baro_alt is measured from (the first SCALED_PRESSURE seen)."""

    press_hpa: float
    temp_k: float


def pressure_altitude(press_hpa: float, ref: BaroReference) -> float:
    """Hypsometric altitude (m) of press_hpa above the reference level."""
    return ref.temp_k / ISA_LAPSE_K_PER_M * (1.0 - (press_hpa / ref.press_hpa) ** ISA_EXPONENT)


def baro_reference(msg: Any) -> BaroReference:
    return BaroReference(press_hpa=float(msg.press_abs), temp_k=msg.temperature / 100.0 + 273.15)


def message_to_channels(msg: Any, baro_ref: BaroReference | None = None) -> dict[str, float]:
    """QUAD channels carried by one MAVLink message, in schema units. Unknown messages map to {}."""
    kind = msg.get_type()
    if kind == "LOCAL_POSITION_NED":
        return {"pn": msg.x, "pe": msg.y, "alt": -msg.z, "vn": msg.vx, "ve": msg.vy, "vu": -msg.vz}
    if kind == "ATTITUDE":
        return {"roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw,
                "p": msg.rollspeed, "q": msg.pitchspeed, "r": msg.yawspeed}
    if kind in IMU_MESSAGES:
        return {"ax": msg.xacc * MG_TO_MS2, "ay": msg.yacc * MG_TO_MS2, "az": msg.zacc * MG_TO_MS2}
    if kind == "VFR_HUD":
        return {"groundspeed": msg.groundspeed}
    if kind == "SCALED_PRESSURE":
        ref = baro_ref or baro_reference(msg)
        return {"baro_alt": pressure_altitude(msg.press_abs, ref)}
    if kind == "SYS_STATUS":
        if msg.voltage_battery == UINT16_UNKNOWN:
            return {}
        return {"battery_v": msg.voltage_battery / 1000.0}
    return {}


@dataclass
class BridgeState:
    """Latest value of every QUAD channel, assembled from whatever messages have arrived."""

    channels: dict[str, float] = field(default_factory=dict)
    baro_ref: BaroReference | None = None

    def update(self, msg: Any) -> None:
        if msg.get_type() == "SCALED_PRESSURE" and self.baro_ref is None:
            self.baro_ref = baro_reference(msg)
        mapped = message_to_channels(msg, self.baro_ref)
        self.channels.update({name: float(value) for name, value in mapped.items()})

    def missing(self) -> list[str]:
        return [c for c in QUAD.channels if c != "groundspeed" and c not in self.channels]

    def ready(self) -> bool:
        return not self.missing()

    def snapshot(self) -> dict[str, float]:
        """One full QUAD channel dict. Call only when ready()."""
        channels = dict(self.channels)
        channels.setdefault("groundspeed", math.hypot(channels["vn"], channels["ve"]))
        return {c: channels[c] for c in QUAD.channels}


def request_streams(conn: Any, target_system: int, target_component: int, rate_hz: float) -> None:
    """Ask the autopilot to stream every message the bridge uses at rate_hz (MAVLink 2 and legacy requests)."""
    from pymavlink import mavutil

    mav = mavutil.mavlink
    interval_us = int(1e6 / rate_hz)
    for name in STREAMED_MESSAGES:
        msg_id = getattr(mav, f"MAVLINK_MSG_ID_{name}")
        conn.mav.command_long_send(target_system, target_component, mav.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                   msg_id, interval_us, 0, 0, 0, 0, 0)
    conn.mav.request_data_stream_send(target_system, target_component, mav.MAV_DATA_STREAM_ALL,
                                      max(1, int(round(rate_hz))), 1)


@dataclass
class SitlSource:
    """Source protocol implementation for a live MAVLink feed. Constructing it does not open a socket."""

    udp_port: int = 14550
    rate_hz: float = 20.0
    host: str = "0.0.0.0"
    timeout_s: float = 30.0          # end the stream after this long without any MAVLink message
    request_rates: bool = True       # send SET_MESSAGE_INTERVAL on the first vehicle heartbeat
    seq_start: int = 0
    schema: ChannelSchema = field(default=QUAD, init=False)
    vehicle_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.vehicle_id = f"sitl-udp-{self.udp_port}"

    def frames(self) -> Iterator[Frame]:
        from pymavlink import mavutil

        conn = mavutil.mavlink_connection(f"udpin:{self.host}:{self.udp_port}", source_system=255,
                                          dialect="ardupilotmega")
        try:
            yield from self._stream(conn)
        finally:
            conn.close()

    def _stream(self, conn: Any) -> Iterator[Frame]:
        from pymavlink import mavutil

        state = BridgeState()
        period = 1.0 / self.rate_hz
        start = last_rx = next_emit = time.monotonic()
        t0: float | None = None
        index = 0
        requested = not self.request_rates
        warned = False
        while True:
            msg = conn.recv_match(blocking=True, timeout=max(0.0, next_emit - time.monotonic()))
            now = time.monotonic()
            if msg is not None and msg.get_type() != "BAD_DATA":
                last_rx = now
                is_vehicle_heartbeat = msg.get_type() == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS
                if not requested and is_vehicle_heartbeat:
                    request_streams(conn, msg.get_srcSystem(), msg.get_srcComponent(), self.rate_hz)
                    requested = True
                state.update(msg)
            if now - last_rx > self.timeout_s:
                log.warning("no MAVLink on udp:%s for %.0f s, ending stream", self.udp_port, self.timeout_s)
                return
            if now < next_emit:
                continue
            next_emit += period
            if next_emit < now:  # stalled: restart the cadence instead of bursting to catch up
                next_emit = now + period
            if not state.ready():
                if not warned and now - start > 5.0:
                    log.warning("SITL bridge still waiting for channels: %s", ", ".join(state.missing()))
                    warned = True
                continue
            if t0 is None:
                t0 = now
            yield Frame(self.vehicle_id, QUAD.schema_id, seq_at(index, self.seq_start), now - t0, state.snapshot())
            index += 1

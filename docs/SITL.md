# Live ArduPilot SITL → Valid-But-Wrong

`vbw.sources.mavlink_bridge.SitlSource` listens for MAVLink on UDP, keeps the latest value of every QUAD channel and
emits QUAD frames at a fixed rate (20 Hz by default). In the scenario catalog it is `sitl:udp:14550`; any port works
via `sitl:udp:<port>`.

```
ArduCopter SITL ──tcp:5760──▶ MAVProxy ──udp:127.0.0.1:14550──▶ SitlSource ──▶ link codec ──▶ detector
                                 ▲
                         you type flight commands here
```

## What the bridge needs from the autopilot

| MAVLink message | QUAD channels | conversion |
|---|---|---|
| `LOCAL_POSITION_NED` | pn, pe, alt, vn, ve, vu | alt = −z, vu = −vz (NED down → up) |
| `ATTITUDE` | roll, pitch, yaw, p, q, r | rad, rad/s (body rates) |
| `SCALED_IMU` or `RAW_IMU` | ax, ay, az | mG × 9.80665/1000 → m/s², body FRD; level rest gives az ≈ −9.81 |
| `VFR_HUD` | groundspeed | m/s (falls back to hypot(vn, ve) until the first VFR_HUD) |
| `SCALED_PRESSURE` | baro_alt | hypsometric altitude above the first pressure reading |
| `SYS_STATUS` | battery_v | mV → V |

Frames start once every channel has been seen. If a message never arrives, the bridge logs
`SITL bridge still waiting for channels: ...` after 5 s. The stream ends after 30 s with no MAVLink at all.

**Stream rate matters.** The bridge does sample-and-hold. If the autopilot streams position at MAVProxy's default
4 Hz while the bridge emits 20 Hz, consecutive frames repeat the same position while velocity is non-zero. That is
physically inconsistent and the detector will (correctly) complain. On the first vehicle heartbeat the bridge sends
`SET_MESSAGE_INTERVAL` for each message above at 20 Hz. MAVProxy can still override that, so also start MAVProxy
with `--streamrate=20` (below).

## Option A: native install (macOS or Ubuntu)

ArduPilot's own setup scripts install the toolchain and MAVProxy. Allow 15–30 minutes the first time.

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ~/ardupilot
Tools/environment_install/install-prereqs-mac.sh -y      # macOS
# Tools/environment_install/install-prereqs-ubuntu.sh -y  # Ubuntu
# then open a new shell (or source the profile file the script says it changed) so PATH picks up the tools

# Build and start a copter. -w wipes params on first run. MAVProxy forwards to the bridge on 14550.
Tools/autotest/sim_vehicle.py -v ArduCopter -w --console \
    --out=udp:127.0.0.1:14550 \
    --mavproxy-args="--streamrate=20"
```

## Option B: Docker

ArduPilot ships a `Dockerfile` for its build environment at the repo root. Build it once, then build and run SITL
inside it. The container sends MAVLink out to the bridge on the host.

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ~/ardupilot
docker build . -t ardupilot --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
docker run --rm -it -v "$(pwd):/ardupilot" -u "$(id -u):$(id -g)" ardupilot:latest bash

# inside the container:
./waf configure --board sitl && ./waf copter
Tools/autotest/sim_vehicle.py -v ArduCopter -w --no-rebuild \
    --out=udp:host.docker.internal:14550 \
    --mavproxy-args="--streamrate=20"
```

On Linux, `host.docker.internal` needs `--add-host=host.docker.internal:host-gateway` on `docker run`
(or run the container with `--network host` and use `--out=udp:127.0.0.1:14550`).

## Fly something

Type these in the MAVProxy console once the EKF reports it is using GPS (roughly 30 s after start):

```
mode guided
arm throttle
takeoff 20
mode circle      # orbit, gives non-trivial attitude and acceleration
mode rtl         # fly home and land
```

Any mission works too (`wp load <file>` then `mode auto`).

## Check the bridge on its own

```bash
uv run python -c "
from vbw.sources.registry import make_source
for i, f in zip(range(40), make_source('sitl:udp:14550').frames()):
    c = f.channels
    print(f'{f.seq:5d} t={f.t:6.2f} alt={c[\"alt\"]:6.2f} baro={c[\"baro_alt\"]:6.2f} az={c[\"az\"]:6.2f} V={c[\"battery_v\"]:5.2f}')
"
```

On the pad you should see alt ≈ 0 and az ≈ −9.8; battery_v is SITL's simulated pack (`SIM_BATT_VOLTAGE`). Then pick
`ArduPilot SITL · live MAVLink (UDP 14550)` in the console, or POST `{"action": "start", "scenario": "sitl:udp:14550"}`
to `/api/control`.

## Troubleshooting

- **Nothing arrives / `Address already in use`.** QGroundControl and Mission Planner also listen on UDP 14550. Close
  them, or point SITL at another port (`--out=udp:127.0.0.1:14551`) and use the scenario `sitl:udp:14551`.
- **Bridge waits forever for some channels.** That stream group is off or slow. In MAVProxy run `set streamrate 20`,
  or set the ArduCopter stream-group params directly: `SR0_POSITION` (LOCAL_POSITION_NED), `SR0_EXTRA1` (ATTITUDE),
  `SR0_EXTRA2` (VFR_HUD), `SR0_RAW_SENS` (IMU, SCALED_PRESSURE), `SR0_EXT_STAT` (SYS_STATUS), e.g.
  `param set SR0_RAW_SENS 20`.
- **Position looks like a staircase.** The stream rate is below the bridge rate; see "Stream rate matters".
- **macOS firewall prompt.** Allow incoming connections for the Python interpreter, or bind to 127.0.0.1 only:
  `SitlSource(udp_port=14550, host="127.0.0.1")`.

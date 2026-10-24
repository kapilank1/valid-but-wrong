"""Scenario catalog: every replayable or live telemetry source, addressed by a scenario id (docs/CONTRACT.md).

Ids: falcon9:<slug> | quad:<mission> | smap:<chan_id> | msl:<chan_id> | sitl:udp:<port>

The hero launch is thaicom-8: on the data-quality checks it has the most self-consistent kinematics of all 31
launches (1-s |d(alt)/dt - vz| p95 16.4 m/s vs >= 20.7 m/s for every other launch, |speed - hypot(vz, vx)| median
0.14 m/s, the lowest), runs 546 s, and carries every ascent event (maxq, meco, ses1, seco1, throttle-down).

list_scenarios() is cheap: it reads metadata only (no simulation, no .npy loads, no sockets).
"""

from __future__ import annotations

from vbw.schema import FALCON9, QUAD, SPACECRAFT, Source
from vbw.sources.falcon9 import RATE_HZ as FALCON9_RATE_HZ
from vbw.sources.falcon9 import Falcon9Source, falcon9_index
from vbw.sources.mavlink_bridge import SitlSource
from vbw.sources.quad_sim import MISSIONS, QuadSim, mission_plan
from vbw.sources.spacecraft import SpacecraftSource, channel_labels

HERO_SLUG = "thaicom-8"
QUAD_RATE_HZ = 20.0
SPACECRAFT_RATE_HZ = 10.0
SITL_PORT = 14550

QUAD_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "survey": ("Quadrotor · Survey grid",
               "Simulated 1.6 kg quad flying a 4-line lawnmower survey at 30 m in gusty wind, then landing."),
    "delivery": ("Quadrotor · Delivery run",
                 "Simulated quad cruising 200 m at 40 m, descending to 8 m for a drop, and flying home."),
    "hover": ("Quadrotor · Station keeping",
              "Simulated quad holding position at 12 m in strong gusts while scanning its heading."),
}

# Test channels with labeled anomalies that show up clearly in the value itself (anomaly-window mean shift of
# 1.5-101 outside-window std devs, >= 15 distinct levels so they plot as telemetry, not a binary flag), a mix of
# point and contextual anomalies, plus P-1, the canonical Telemanom example.
SPACECRAFT_CHANNELS: tuple[tuple[str, str], ...] = (
    ("smap", "P-1"),
    ("smap", "D-5"),
    ("smap", "E-7"),
    ("smap", "T-3"),
    ("msl", "M-7"),
    ("msl", "F-5"),
    ("msl", "P-11"),
    ("msl", "T-9"),
)

STAGE_NAMES = {1: "first-stage (booster)", 2: "second-stage"}


def _falcon9_scenario(launch: dict) -> dict:
    events = {name: float(t) for name, t in launch.get("events", {}).items()}
    marks = ", ".join(f"{name} {t:.0f} s" for name, t in sorted(events.items(), key=lambda kv: kv[1]))
    stage = STAGE_NAMES.get(launch.get("stage"), "webcast")
    return {
        "id": f"falcon9:{launch['slug']}",
        "label": f"Falcon 9 · {launch['mission_name']}",
        "schema": FALCON9.name,
        "vehicle_id": f"falcon9-{launch['slug']}",
        "rate_hz": FALCON9_RATE_HZ,
        "duration_s": float(launch["duration_s"]),
        "events": events,
        "description": f"Real {stage} webcast telemetry, {launch['duration_s']:.0f} s at 10 Hz. Events: {marks}.",
        "hero": launch["slug"] == HERO_SLUG,
    }


def _quad_scenario(mission: str) -> dict:
    plan = mission_plan(mission)
    label, description = QUAD_DESCRIPTIONS[mission]
    return {
        "id": f"quad:{mission}",
        "label": label,
        "schema": QUAD.name,
        "vehicle_id": f"quad-{mission}",
        "rate_hz": QUAD_RATE_HZ,
        "duration_s": round(plan.duration_s, 3),
        "events": dict(plan.events),
        "description": description,
        "hero": False,
    }


def _spacecraft_scenario(dataset: str, chan_id: str) -> dict:
    labels = channel_labels()[chan_id]
    sequences = [[a, b] for a, b in labels.anomaly_sequences]
    t_ranges = [[a / SPACECRAFT_RATE_HZ, b / SPACECRAFT_RATE_HZ] for a, b in sequences]
    kinds = ", ".join(sorted(set(labels.classes)))
    return {
        "id": f"{dataset}:{chan_id}",
        "label": f"{dataset.upper()} · {chan_id}",
        "schema": SPACECRAFT.name,
        "vehicle_id": f"{dataset}-{chan_id}",
        "rate_hz": SPACECRAFT_RATE_HZ,
        "duration_s": labels.num_values / SPACECRAFT_RATE_HZ,
        "events": {f"anomaly_{i + 1}": t0 for i, (t0, _) in enumerate(t_ranges)},
        "description": (f"Real NASA {dataset.upper()} channel {chan_id} (normalized), replayed at 10 Hz with "
                        f"{len(sequences)} labeled {kinds} anomal{'y' if len(sequences) == 1 else 'ies'}."),
        "anomaly_sequences": sequences,
        "anomaly_t_ranges": t_ranges,
        "hero": False,
    }


def _sitl_scenario(port: int) -> dict:
    return {
        "id": f"sitl:udp:{port}",
        "label": f"ArduPilot SITL · live MAVLink (UDP {port})",
        "schema": QUAD.name,
        "vehicle_id": f"sitl-udp-{port}",
        "rate_hz": QUAD_RATE_HZ,
        "duration_s": None,
        "events": {},
        "description": f"Live ArduPilot SITL (or vehicle) on udp:{port}, resampled to QUAD frames. See docs/SITL.md.",
        "hero": False,
    }


def list_scenarios() -> list[dict]:
    """All scenarios, hero launch first, then the other launches, quad missions, spacecraft channels, SITL."""
    launches = sorted(falcon9_index(), key=lambda launch: launch["slug"] != HERO_SLUG)
    return (
        [_falcon9_scenario(launch) for launch in launches]
        + [_quad_scenario(mission) for mission in MISSIONS]
        + [_spacecraft_scenario(dataset, chan) for dataset, chan in SPACECRAFT_CHANNELS]
        + [_sitl_scenario(SITL_PORT)]
    )


def make_source(scenario_id: str, seed: int | None = None) -> Source:
    """Build the source for a scenario id. Accepts any launch slug, mission, channel or port, not just listed ones.

    seed only affects stochastic sources (Falcon 9 sensor noise, quad wind + sensor noise); spacecraft replays
    and SITL are deterministic / live.
    """
    kind, _, rest = scenario_id.partition(":")
    if not rest:
        raise ValueError(f"bad scenario id {scenario_id!r}")
    if kind == "falcon9":
        return Falcon9Source(rest, seed=seed)
    if kind == "quad":
        return QuadSim(rest, seed=seed, rate_hz=QUAD_RATE_HZ)
    if kind in ("smap", "msl"):
        return SpacecraftSource(kind, rest, rate_hz=SPACECRAFT_RATE_HZ)
    if kind == "sitl":
        transport, _, port = rest.partition(":")
        if transport != "udp" or not port.isdigit():
            raise ValueError(f"bad SITL scenario id {scenario_id!r}; expected sitl:udp:<port>")
        return SitlSource(udp_port=int(port), rate_hz=QUAD_RATE_HZ)
    raise ValueError(f"unknown scenario kind {kind!r} in {scenario_id!r}")

import itertools

import pytest

from vbw.schema import SCHEMAS_BY_NAME
from vbw.sources.falcon9 import falcon9_index
from vbw.sources.mavlink_bridge import SitlSource
from vbw.sources.registry import HERO_SLUG, list_scenarios, make_source

REQUIRED_KEYS = {"id", "label", "schema", "vehicle_id", "rate_hz", "duration_s", "events", "description"}


@pytest.fixture(scope="module")
def scenarios():
    return list_scenarios()


def by_kind(scenarios, kind):
    return [s for s in scenarios if s["id"].split(":")[0] == kind]


def test_catalog_shape(scenarios):
    assert len({s["id"] for s in scenarios}) == len(scenarios)
    for s in scenarios:
        assert REQUIRED_KEYS <= set(s), s["id"]
        assert s["schema"] in SCHEMAS_BY_NAME
        assert isinstance(s["events"], dict)
    assert len(by_kind(scenarios, "falcon9")) == len(falcon9_index()) == 31
    assert len(by_kind(scenarios, "quad")) == 3
    assert len(by_kind(scenarios, "smap")) + len(by_kind(scenarios, "msl")) == 8
    assert [s["id"] for s in by_kind(scenarios, "sitl")] == ["sitl:udp:14550"]


def test_hero_launch_is_first_and_unique(scenarios):
    heroes = [s for s in scenarios if s.get("hero")]
    assert len(heroes) == 1 and scenarios[0] is heroes[0]
    assert scenarios[0]["id"] == f"falcon9:{HERO_SLUG}"
    assert {"maxq", "meco"} <= set(scenarios[0]["events"])
    assert scenarios[0]["label"].startswith("Falcon 9 · ")


def first_frames(source, n=3):
    return list(itertools.islice(source.frames(), n))


def assert_matches_scenario(source, scenario):
    assert source.vehicle_id == scenario["vehicle_id"]
    assert source.schema.name == scenario["schema"]
    assert source.rate_hz == scenario["rate_hz"]


def test_every_falcon9_scenario_round_trips(scenarios):
    for scenario in by_kind(scenarios, "falcon9"):
        source = make_source(scenario["id"], seed=0)
        assert_matches_scenario(source, scenario)
        assert source.events == scenario["events"] and source.duration_s == scenario["duration_s"]
    frames = first_frames(make_source(scenarios[0]["id"], seed=0))
    assert tuple(frames[0].channels) == SCHEMAS_BY_NAME["falcon9"].channels


def test_quad_scenarios_round_trip(scenarios):
    for scenario in by_kind(scenarios, "quad"):
        source = make_source(scenario["id"], seed=1)
        assert_matches_scenario(source, scenario)
        assert source.duration_s == pytest.approx(scenario["duration_s"], abs=1e-3)
        assert source.events == scenario["events"]
        frames = first_frames(source)
        assert tuple(frames[0].channels) == SCHEMAS_BY_NAME["quad"].channels


def test_spacecraft_scenarios_round_trip_with_anomaly_ranges(scenarios):
    spacecraft = by_kind(scenarios, "smap") + by_kind(scenarios, "msl")
    for scenario in spacecraft:
        source = make_source(scenario["id"])
        assert_matches_scenario(source, scenario)
        assert source.anomaly_t_ranges == scenario["anomaly_t_ranges"]
        assert source.anomaly_sequences == scenario["anomaly_sequences"]
        assert len(source.anomaly_sequences) >= 1
        assert source.duration_s == pytest.approx(scenario["duration_s"])
        assert first_frames(source, 1)[0].channels.keys() == {"value"}


def test_sitl_scenario_constructs_without_listening(scenarios):
    scenario = by_kind(scenarios, "sitl")[0]
    source = make_source(scenario["id"])
    assert isinstance(source, SitlSource) and source.udp_port == 14550
    assert_matches_scenario(source, scenario)
    assert scenario["duration_s"] is None
    assert make_source("sitl:udp:14551").udp_port == 14551


def test_unlisted_but_valid_ids_work():
    assert make_source("smap:A-1").vehicle_id == "smap-A-1"
    assert make_source("falcon9:zuma", seed=0).vehicle_id == "falcon9-zuma"


@pytest.mark.parametrize("bad", ["nope", "rocket:abc", "sitl:tcp:5760", "sitl:udp:abc"])
def test_bad_ids_raise(bad):
    with pytest.raises(ValueError):
        make_source(bad)


@pytest.mark.parametrize("bad", ["quad:barrel-roll", "falcon9:not-a-launch"])
def test_unknown_names_raise(bad):
    with pytest.raises(KeyError):
        make_source(bad)

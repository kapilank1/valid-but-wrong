from vbw.detector.splits import holdout_slugs
from vbw.fleet.vehicle import ATTACK_ONSET_FRACTION, FLEET_ATTACKS, VehicleSpec, VehicleStream, fleet_specs
from vbw.fleet.worker import detector_factory

SES9 = "ses-9"


def fly(stream: VehicleStream):
    """Step until the current flight ends; return (verdicts, result)."""
    verdicts = []
    while True:
        verdict, result = stream.step()
        if result is not None:
            return verdicts, result
        verdicts.append(verdict)


def test_fleet_specs_are_deterministic_and_rotate_launches():
    slugs = holdout_slugs()
    a, b = fleet_specs(100, slugs, 0.2), fleet_specs(100, slugs, 0.2)
    assert a == b
    assert [s.slug for s in a[: len(slugs)]] == slugs
    assert len({s.vehicle_id for s in a}) == 100


def test_attack_fraction_is_respected():
    specs = fleet_specs(5000, holdout_slugs(), 0.1)
    attacked = [s for s in specs if s.attack]
    assert 0.08 < len(attacked) / len(specs) < 0.12
    assert {s.attack for s in attacked} == set(FLEET_ATTACKS)
    assert not any(s.attack for s in fleet_specs(200, holdout_slugs(), 0.0))


def test_bias_attack_passes_the_link_and_is_detected_after_onset():
    stream = VehicleStream(VehicleSpec("f9-test", SES9, seed=1, attack="bias"), detector_factory("base"))
    onset = ATTACK_ONSET_FRACTION * 544.9                # SES-9 lasts 544.9 s
    verdicts, result = fly(stream)
    assert all(v.l0.ok for v in verdicts)               # the forgery is CRC-valid and in sequence
    assert all(v.vehicle_id == "f9-test" for v in verdicts)
    assert not any(v.alert for v in verdicts if v.t < onset)
    assert result.attacked and result.detected and not result.false_alert
    assert 0 <= result.latency_s < 5


def test_clean_holdout_flight_has_no_false_alert_and_restarts():
    stream = VehicleStream(VehicleSpec("f9-clean", SES9, seed=2, attack=None), detector_factory("base"))
    verdicts, result = fly(stream)
    assert not result.attacked and not result.false_alert and not result.detected
    assert not any(v.alert for v in verdicts)
    assert stream.flights == 1                          # the next flight already started
    verdict, _ = stream.step()
    assert verdict.l0.ok                                # new flight: link monitor was reset with it

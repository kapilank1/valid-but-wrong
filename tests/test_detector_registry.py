from vbw.detector.alerting import AlertConfig
from vbw.detector.registry import list_detectors


def test_list_detectors_reports_the_calibrated_threshold_per_schema():
    detectors = list_detectors()
    assert {d["name"] for d in detectors} == {"base", "hardened"}
    for d in detectors:
        assert d["thresholds"]["falcon9"] == AlertConfig.load("falcon9").tau == d["threshold"]
        assert d["thresholds"]["quad"] == AlertConfig.load("quad").tau

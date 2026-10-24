from vbw.pipeline.callouts import finding, spoken_line
from vbw.schema import L0Result, Verdict, Violation


def _verdict(l1_score: float, l2_score: float, with_violation: bool = True) -> Verdict:
    v = Verdict("f9", 1, 10.0, L0Result(), l1_score=l1_score, l2_score=l2_score, score=0.9, alert=True)
    if with_violation:
        v.l1_violations = [Violation("q_vs_density", ["q", "altitude"], 7.5, 2.4, "q off")]
    v.l2_residuals = {"altitude": 50.0, "speed": -0.4}
    return v


def test_physics_violation_is_spoken_when_l1_leads():
    assert finding(_verdict(0.9, 0.3), "falcon9") == "Dynamic pressure disagrees with altitude."


def test_learned_layer_is_spoken_when_it_scored_higher_like_the_written_explanation():
    assert finding(_verdict(0.4, 0.9), "falcon9") == "Altitude departing from predicted track."


def test_spoken_line_has_station_finding_and_confidence():
    line = spoken_line(_verdict(0.9, 0.3), "falcon9")
    assert line.startswith("Flight, Guidance. Dynamic pressure disagrees with altitude.")
    assert line.endswith("Confidence 90 percent.")

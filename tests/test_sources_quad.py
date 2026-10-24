import math

import numpy as np
import pandas as pd
import pytest

from vbw.schema import QUAD
from vbw.sources.quad_sim import (
    BARO_BIAS_SIGMA_M,
    GRAVITY,
    MISSIONS,
    SENSOR_SIGMA,
    QuadSim,
    body_rates_from_euler_rates,
    euler_rates,
    mission_plan,
    rotation_body_to_ned,
    wrap_angle,
)

DT = 1.0 / 20.0
WINDOW = 20  # 1 s at 20 Hz


def frames_dataframe(frames) -> pd.DataFrame:
    return pd.DataFrame([{"t": f.t, **f.channels} for f in frames])


@pytest.fixture(scope="module")
def survey():
    """First 60 s of a survey: spool-up, takeoff, transit and the start of the first survey line."""
    sim = QuadSim("survey", seed=7, duration_s=60.0)
    return sim, sim.truth_dataframe(), frames_dataframe(sim.frames())


def window_position_error(df: pd.DataFrame, axis: str, vel: str) -> np.ndarray:
    """Position change over 1 s minus the trapezoidal integral of velocity over the same window."""
    p, v = df[axis].to_numpy(), df[vel].to_numpy()
    integral = np.convolve(0.5 * (v[1:] + v[:-1]) * DT, np.ones(WINDOW), "valid")
    return (p[WINDOW:] - p[:-WINDOW]) - integral


def ned_acceleration(df: pd.DataFrame) -> np.ndarray:
    """R(attitude) * specific force + g, per row, in NED."""
    out = np.empty((len(df), 3))
    for i, row in enumerate(df.itertuples(index=False)):
        R = rotation_body_to_ned(row.roll, row.pitch, row.yaw)
        out[i] = R @ np.array([row.ax, row.ay, row.az]) + np.array([0.0, 0.0, GRAVITY])
    return out


# --------------------------------------------------------------------------- missions & determinism


@pytest.mark.parametrize("mission", sorted(MISSIONS))
def test_missions_last_two_to_four_minutes_with_ordered_events(mission):
    plan = mission_plan(mission)
    assert 120.0 <= plan.duration_s <= 240.0
    times = list(plan.events.values())
    assert times == sorted(times) and {"takeoff", "land", "disarm"} <= set(plan.events)


def test_same_seed_is_deterministic_and_seeds_differ():
    a = frames_dataframe(QuadSim("delivery", seed=3, duration_s=15.0).frames())
    b = frames_dataframe(QuadSim("delivery", seed=3, duration_s=15.0).frames())
    c = frames_dataframe(QuadSim("delivery", seed=4, duration_s=15.0).frames())
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["pn"], c["pn"])


def test_frames_have_every_schema_channel_and_regular_time(survey):
    sim, truth, noisy = survey
    frames = list(QuadSim("survey", seed=7, duration_s=2.0).frames())
    assert sim.vehicle_id == "quad-survey" and sim.schema is QUAD and sim.rate_hz == 20.0
    assert all(tuple(f.channels) == QUAD.channels and f.schema_id == QUAD.schema_id for f in frames)
    assert [f.seq for f in frames[:3]] == [0, 1, 2]
    assert np.allclose(np.diff(noisy["t"]), DT) and len(noisy) == len(truth) == 1201
    assert np.isfinite(noisy[list(QUAD.channels)].to_numpy()).all()


def test_seq_wraps_at_uint16():
    frames = list(QuadSim("hover", seed=0, duration_s=0.5, seq_start=65533).frames())
    assert [f.seq for f in frames[:5]] == [65533, 65534, 65535, 0, 1]


# --------------------------------------------------------------------------- physics consistency


def test_position_integrates_velocity(survey):
    _, truth, noisy = survey
    for axis, vel in (("pn", "vn"), ("pe", "ve"), ("alt", "vu")):
        assert np.abs(window_position_error(truth, axis, vel)).max() < 0.01, axis
        # With noise, the 1-s error is dominated by two independent position samples.
        expected = math.sqrt(2) * SENSOR_SIGMA[axis]
        assert np.std(window_position_error(noisy, axis, vel)) == pytest.approx(expected, rel=0.25), axis


def test_velocity_integrates_rotated_specific_force_plus_gravity(survey):
    _, truth, noisy = survey
    for df, tolerance in ((truth, 0.02), (noisy, 0.2)):
        acc = ned_acceleration(df)
        vel_ned = np.column_stack([df["vn"], df["ve"], -df["vu"]])
        integral = np.array([
            np.convolve(0.5 * (acc[1:, k] + acc[:-1, k]) * DT, np.ones(WINDOW), "valid") for k in range(3)
        ]).T
        error = (vel_ned[WINDOW:] - vel_ned[:-WINDOW]) - integral
        flying = df["alt"].to_numpy()[WINDOW:] > 0.5  # skip gear-contact transients on the pad
        assert np.percentile(np.abs(error[flying]), 99) < tolerance


def test_body_rates_match_euler_derivatives(survey):
    _, truth, _ = survey
    euler = truth[["roll", "pitch", "yaw"]].to_numpy().copy()
    euler[:, 2] = np.unwrap(euler[:, 2])
    rates = np.array([euler_rates(r.roll, r.pitch, np.array([r.p, r.q, r.r])) for r in truth.itertuples()])
    error = np.diff(euler, axis=0) / DT - 0.5 * (rates[1:] + rates[:-1])
    assert np.abs(error).max() < 0.01
    assert np.abs(truth[["p", "q", "r"]].to_numpy()).max() > 0.05  # the vehicle really does rotate


def test_euler_kinematics_round_trip():
    euler_dot = np.array([0.3, -0.2, 0.5])
    rates = body_rates_from_euler_rates(0.2, -0.3, euler_dot)
    np.testing.assert_allclose(euler_rates(0.2, -0.3, rates), euler_dot, atol=1e-12)


def test_groundspeed_is_hypot_of_reported_velocity(survey):
    _, truth, noisy = survey
    for df in (truth, noisy):
        np.testing.assert_allclose(df["groundspeed"], np.hypot(df["vn"], df["ve"]), rtol=0, atol=1e-12)


def test_accelerometer_reads_minus_g_at_rest_and_hover(survey):
    _, truth, noisy = survey
    on_pad = noisy["t"] < 0.5
    assert noisy.loc[on_pad, "az"].mean() == pytest.approx(-GRAVITY, abs=0.15)
    hovering = (truth["t"] > 18.0) & (truth["t"] < 21.0)  # top of the climb, before the transit leg
    assert truth.loc[hovering, "az"].mean() == pytest.approx(-GRAVITY, abs=0.5)


def test_baro_tracks_alt_with_a_slow_bias(survey):
    _, truth, noisy = survey
    offset = noisy["baro_alt"] - truth["alt"]
    assert np.abs(offset).max() < 4 * BARO_BIAS_SIGMA_M + 0.5
    smoothed = offset.rolling(40, center=True).mean().dropna()
    assert np.abs(np.diff(smoothed)).max() < 0.05  # bias moves slowly; the rest is white noise


def test_battery_sags_under_thrust_and_drains(survey):
    _, truth, _ = survey
    resting = truth.loc[truth["t"] < 0.2, "battery_v"].mean()
    climbing = truth.loc[(truth["t"] > 5) & (truth["t"] < 10), "battery_v"].mean()
    assert resting - climbing > 0.5
    assert truth["soc"].iloc[-1] < truth["soc"].iloc[0]
    sag = truth["battery_v"].diff()
    assert np.corrcoef(truth["thrust_n"].diff()[1:], sag[1:])[0, 1] < -0.5


def test_full_hover_mission_lands_and_disarms():
    truth = QuadSim("hover", seed=11).truth_dataframe()
    end = truth.iloc[-1]
    assert abs(end["alt"]) < 0.05 and end["groundspeed"] < 0.05 and end["thrust_n"] == pytest.approx(0, abs=1e-3)
    assert abs(end["roll"]) < 0.01 and abs(end["pitch"]) < 0.01
    assert end["az"] == pytest.approx(-GRAVITY, abs=0.1)
    assert truth["alt"].max() == pytest.approx(12.0, abs=0.5)
    assert (np.abs(truth["yaw"]) <= math.pi).all()


def test_wrap_angle():
    assert wrap_angle(math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi) == pytest.approx(math.pi)
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)

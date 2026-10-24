import numpy as np
import pytest

from vbw.schema import FALCON9
from vbw.sources.falcon9 import NOISE_SIGMA, Q_RELATIVE_SIGMA, Falcon9Source, falcon9_index

SLUG = "thaicom-8"


def as_array(frames):
    return np.array([[f.channels[c] for c in FALCON9.channels] for f in frames])


@pytest.fixture(scope="module")
def truth():
    return Falcon9Source(SLUG, seed=0).truth_dataframe()


def test_frames_follow_csv_rows_and_schema(truth):
    source = Falcon9Source(SLUG, seed=1)
    frames = list(source.frames())
    assert len(frames) == len(truth)
    assert source.vehicle_id == f"falcon9-{SLUG}" and source.rate_hz == 10.0 and source.schema is FALCON9
    assert [f.seq for f in frames[:3]] == [0, 1, 2]
    np.testing.assert_allclose([f.t for f in frames], truth["t"].to_numpy())
    for f in frames[:: len(frames) // 10]:
        assert tuple(f.channels) == FALCON9.channels
        assert f.schema_id == FALCON9.schema_id
        assert all(np.isfinite(v) for v in f.channels.values())


def test_same_seed_same_frames_different_seed_differs():
    a = as_array(Falcon9Source(SLUG, seed=5).frames())
    b = as_array(Falcon9Source(SLUG, seed=5).frames())
    c = as_array(Falcon9Source(SLUG, seed=6).frames())
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_frames_are_repeatable_without_explicit_seed():
    source = Falcon9Source(SLUG)
    np.testing.assert_array_equal(as_array(source.frames()), as_array(source.frames()))


def test_noise_off_replays_truth_exactly(truth):
    clean = as_array(Falcon9Source(SLUG, noise=False).frames())
    np.testing.assert_array_equal(clean, truth[list(FALCON9.channels)].to_numpy())


def test_noise_matches_documented_sigmas(truth):
    noisy = as_array(Falcon9Source(SLUG, seed=2).frames())
    clean = truth[list(FALCON9.channels)].to_numpy()
    moving = clean[:, FALCON9.index("speed")] > 1.0  # skip the pad, where speed is clipped at 0
    for name, sigma in NOISE_SIGMA.items():
        i = FALCON9.index(name)
        assert np.std(noisy[moving, i] - clean[moving, i]) == pytest.approx(sigma, rel=0.1), name
    iq = FALCON9.index("q")
    has_q = clean[:, iq] > 100.0
    rel = noisy[has_q, iq] / clean[has_q, iq] - 1.0
    assert np.std(rel) == pytest.approx(Q_RELATIVE_SIGMA, rel=0.1)
    assert (noisy[:, iq] >= 0).all() and (noisy[clean[:, iq] == 0, iq] == 0).all()
    assert (noisy[:, FALCON9.index("speed")] >= 0).all()


def test_seq_wraps_at_uint16():
    frames = list(Falcon9Source(SLUG, seed=0, seq_start=65534).frames())
    assert [f.seq for f in frames[:4]] == [65534, 65535, 0, 1]


def test_events_and_metadata_come_from_index():
    source = Falcon9Source(SLUG, seed=0)
    info = next(launch for launch in falcon9_index() if launch["slug"] == SLUG)
    assert source.events == {k: float(v) for k, v in info["events"].items()}
    assert {"maxq", "meco"} <= set(source.events)
    assert source.mission_name == info["mission_name"] and source.duration_s == info["duration_s"]


def test_unknown_slug_raises():
    with pytest.raises(KeyError):
        Falcon9Source("not-a-launch")

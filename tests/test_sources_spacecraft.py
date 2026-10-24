import numpy as np
import pytest

from vbw.schema import SPACECRAFT
from vbw.sources.spacecraft import SpacecraftSource, channel_labels, load_channel


def test_replays_column_zero_with_index_time():
    source = SpacecraftSource("smap", "P-1", rate_hz=10.0)
    frames = list(source.frames())
    raw = load_channel("P-1")
    assert len(frames) == len(raw) and source.duration_s == len(raw) / 10.0
    np.testing.assert_array_equal([f.channels["value"] for f in frames], raw[:, 0])
    assert [f.t for f in frames[:3]] == [0.0, 0.1, 0.2]
    assert all(tuple(f.channels) == SPACECRAFT.channels for f in frames[:10])
    assert source.vehicle_id == "smap-P-1" and frames[0].schema_id == SPACECRAFT.schema_id


def test_anomaly_labels_are_sorted_and_converted_to_time():
    source = SpacecraftSource("smap", "P-1", rate_hz=10.0)
    assert source.anomaly_sequences == [[2149, 2349], [3539, 3779], [4536, 4844]]
    assert source.anomaly_t_ranges == [[214.9, 234.9], [353.9, 377.9], [453.6, 484.4]]
    assert source.events == {"anomaly_1": 214.9, "anomaly_2": 353.9, "anomaly_3": 453.6}


def test_rate_changes_time_but_not_values():
    slow = SpacecraftSource("msl", "T-9", rate_hz=2.0)
    assert slow.anomaly_t_ranges[0] == [780 / 2.0, 810 / 2.0]
    assert list(slow.frames())[1].t == 0.5


def test_dataset_must_match_channel():
    with pytest.raises(ValueError):
        SpacecraftSource("msl", "P-1")
    with pytest.raises(ValueError):
        SpacecraftSource("cassini", "P-1")


def test_train_split_has_no_labels():
    source = SpacecraftSource("smap", "P-1", split="train")
    assert source.anomaly_sequences == [] and source.events == {}
    assert len(source.values) == len(load_channel("P-1", "train"))


def test_label_file_quirks():
    labels = channel_labels()
    assert labels["P-2"].anomaly_sequences == ((5350, 6575),)  # first of the two duplicate rows
    unlabeled = SpacecraftSource("msl", "T-10")                 # no row in the CSV, MSL by column count
    assert unlabeled.anomaly_sequences == []


def test_seq_wraps():
    frames = list(SpacecraftSource("msl", "T-9", seq_start=65535).frames())
    assert [f.seq for f in frames[:3]] == [65535, 0, 1]

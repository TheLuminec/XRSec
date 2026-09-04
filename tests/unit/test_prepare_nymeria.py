"""The Nymeria converter's column selection, timestamp conversion, and
sequence-key parsing -- checked against the real schema (verified from 3
real closed_loop_trajectory.csv files: mean |q| = 0.99999999997 confirms
the world_device quaternion is already this pipeline's x,y,z,w order, and
position needed no unit conversion either -- the first dataset in this
corpus where neither standard trap applies)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prepare_nymeria import convert, convert_session, find_sequences, inspect, parse_sequence_key

pytestmark = pytest.mark.unit

# Column order/names match a real closed_loop_trajectory.csv exactly
# (trimmed to what convert_session reads plus the columns it must ignore).
RAW_COLUMNS = [
    "graph_uid", "tracking_timestamp_us", "utc_timestamp_ns",
    "tx_world_device", "ty_world_device", "tz_world_device",
    "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device",
    "device_linear_velocity_x_device", "quality_score", "geo_available",
    "tx_ecef_device", "qw_ecef_device",
]


def _write_trajectory_csv(path, n=50, dt_us=972, quat_norm=1.0):
    rows = []
    for i in range(n):
        row = {c: 0.0 for c in RAW_COLUMNS}
        row.update({
            "graph_uid": "g", "tracking_timestamp_us": 35_000_000 + dt_us * i,
            "utc_timestamp_ns": 1_688_000_000_000_000_000,
            "tx_world_device": 0.01 * i, "ty_world_device": 1.6, "tz_world_device": 0.0,
            "qx_world_device": 0.0, "qy_world_device": 0.0, "qz_world_device": 0.0,
            "qw_world_device": quat_norm,
            "quality_score": 0.5, "geo_available": 1,
        })
        rows.append(row)
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False)


def test_parses_a_real_sequence_key():
    parsed = parse_sequence_key("20230607_s0_james_johnson_act0_e72nhq")
    assert parsed == ("20230607", "s0", "james_johnson", "0", "e72nhq")


def test_rejects_a_malformed_key():
    assert parse_sequence_key("not_a_sequence_key") is None


def test_quaternion_needs_no_reorder_but_is_still_read_by_name(tmp_path):
    """world_device quaternion columns are already qx,qy,qz,qw -- confirmed
    against real data, not assumed. Still selected by name so a future
    schema change (e.g. a device-frame column reordering) fails loudly via
    the missing-column check rather than silently reading the wrong one."""
    _write_trajectory_csv(tmp_path / "traj.csv")
    session = convert_session(tmp_path / "traj.csv")
    assert (session["UnitQuaternion.w"] == 1.0).all()
    assert (session["UnitQuaternion.x"] == 0.0).all()
    norm = np.linalg.norm(session[["UnitQuaternion.x", "UnitQuaternion.y",
                                    "UnitQuaternion.z", "UnitQuaternion.w"]].to_numpy(), axis=1)
    assert np.allclose(norm, 1.0)


def test_position_needs_no_unit_conversion(tmp_path):
    """ty_world_device = 1.6 in the source must stay 1.6 -- unlike
    who-is-alyx or the cross-app dataset, Nymeria's world_device position
    is already metres."""
    _write_trajectory_csv(tmp_path / "traj.csv")
    session = convert_session(tmp_path / "traj.csv")
    assert np.allclose(session["HmdPosition.y"], 1.6)


def test_timestamp_is_microseconds_and_zero_based(tmp_path):
    """tracking_timestamp_us, not utc_timestamp_ns (which updates far less
    often and is dropped). Divided by 1e6, zero-based per session."""
    _write_trajectory_csv(tmp_path / "traj.csv", n=5, dt_us=972)
    session = convert_session(tmp_path / "traj.csv")
    assert session["SessionTime"].iloc[0] == 0.0
    assert np.allclose(session["SessionTime"].diff().dropna(), 0.000972, atol=1e-9)


def test_max_hz_decimates_but_leaves_low_rate_sources_alone(tmp_path):
    """Native ~1029Hz is by far the most aggressive decimation in the
    corpus (next worst is NJIT's 250Hz at 12:1) -- max_hz exists so the
    stored copy isn't forced all the way down to 20Hz, unlike a naive
    reading of 'nothing samples above 20Hz' might suggest."""
    _write_trajectory_csv(tmp_path / "traj.csv", n=1000, dt_us=972)  # ~1029Hz
    full = convert_session(tmp_path / "traj.csv", max_hz=None)
    decimated = convert_session(tmp_path / "traj.csv", max_hz=60.0)
    assert len(decimated) < len(full)
    assert len(decimated) == pytest.approx(len(full) / (1029 / 60), rel=0.2)

    _write_trajectory_csv(tmp_path / "slow.csv", n=50, dt_us=50_000)  # 20Hz
    untouched = convert_session(tmp_path / "slow.csv", max_hz=60.0)
    assert len(untouched) == 50


def test_find_sequences_reads_the_fake_name_and_act_id(tmp_path):
    seq_dir = tmp_path / "20230607_s0_james_johnson_act0_e72nhq"
    seq_dir.mkdir()
    _write_trajectory_csv(seq_dir / "closed_loop_trajectory.csv")
    found = list(find_sequences(tmp_path))
    assert found == [("james_johnson", "0", seq_dir / "closed_loop_trajectory.csv")]


def test_a_directory_not_matching_the_key_pattern_is_ignored(tmp_path):
    (tmp_path / "not_a_sequence_key").mkdir()
    (tmp_path / "not_a_sequence_key" / "closed_loop_trajectory.csv").write_text("x\n1\n")
    assert list(find_sequences(tmp_path)) == []


def test_convert_writes_one_file_per_activity_with_citation(tmp_path):
    for date_session, name, act in [
        ("20230607_s0", "james_johnson", "0"),
        ("20230607_s0", "james_johnson", "1"),
    ]:
        seq_dir = tmp_path / f"{date_session}_{name}_act{act}_abc123"
        seq_dir.mkdir()
        _write_trajectory_csv(seq_dir / "closed_loop_trajectory.csv")

    out = tmp_path / "out"
    rc = convert(tmp_path, out)
    assert rc == 0
    assert (out / "CITATION.txt").exists()
    assert (out / "james_johnson" / "act0.csv").exists()
    assert (out / "james_johnson" / "act1.csv").exists()


def test_convert_carries_participants_metadata_through_unmodified(tmp_path):
    seq_dir = tmp_path / "20230607_s0_james_johnson_act0_abc123"
    seq_dir.mkdir()
    _write_trajectory_csv(seq_dir / "closed_loop_trajectory.csv")

    metadata = tmp_path / "participants_metadata.csv"
    metadata.write_text("# All names are fake.\ndate,name,height_cm\n20230607,james_johnson,180\n")

    out = tmp_path / "out"
    convert(tmp_path, out, metadata_csv=metadata)
    assert (out / "participants_metadata.csv").read_text() == metadata.read_text()


def test_convert_flags_but_still_writes_a_bad_quaternion_session(tmp_path):
    seq_dir = tmp_path / "20230607_s0_james_johnson_act0_abc123"
    seq_dir.mkdir()
    _write_trajectory_csv(seq_dir / "closed_loop_trajectory.csv", quat_norm=5.0)

    out = tmp_path / "out"
    rc = convert(tmp_path, out)
    assert rc == 1
    assert (out / "james_johnson" / "act0.csv").exists()


def test_inspect_runs_without_writing_anything(tmp_path):
    seq_dir = tmp_path / "20230607_s0_james_johnson_act0_abc123"
    seq_dir.mkdir()
    _write_trajectory_csv(seq_dir / "closed_loop_trajectory.csv")
    rc = inspect(tmp_path)
    assert rc == 0
    assert not (tmp_path / "out").exists()

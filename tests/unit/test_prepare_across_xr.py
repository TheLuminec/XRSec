"""The cross-application-XR converter's column reordering, unit conversion,
timestamp parsing, and session-splitting -- checked against the real schema
(reported by a peer session that could reach the source repo when this one
could not), not the paper's or README's descriptions of it, both of which
turned out to be wrong about the quaternion order."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prepare_across_xr import convert, find_user_csvs, inspect, load_user_frame, split_sessions

pytestmark = pytest.mark.unit

# Column order matches a real file's header exactly, including the
# scalar-first quaternion (head_rot_w, ...) that neither the paper's text
# ("rotations x,y,z,w") nor the README ("rotation x,y,z") got right.
RAW_COLUMNS = [
    "timestamp", "head_pos_x", "head_pos_y", "head_pos_z",
    "head_rot_w", "head_rot_x", "head_rot_y", "head_rot_z",
    "right_hand_pos_x", "right_hand_pos_y", "right_hand_pos_z",
    "right_hand_rot_w", "right_hand_rot_x", "right_hand_rot_y", "right_hand_rot_z",
    "left_hand_pos_x", "left_hand_pos_y", "left_hand_pos_z",
    "left_hand_rot_w", "left_hand_rot_x", "left_hand_rot_y", "left_hand_rot_z",
    "take_id", "user_id", "game_id",
]


def _write_user_csv(path, user_id, games_takes, n=50, dt_ms=11, quat_norm=1.0):
    """games_takes: list of (game_id, take_id) pairs to generate rows for."""
    rows = []
    t0 = pd.Timedelta(minutes=18, seconds=47, milliseconds=969)
    for game_id, take_id in games_takes:
        for i in range(n):
            t = t0 + pd.Timedelta(milliseconds=dt_ms * i)
            row = {c: 0.0 for c in RAW_COLUMNS}
            row.update({
                "timestamp": str(t),
                "head_pos_x": 0.0, "head_pos_y": 161.0, "head_pos_z": 0.0,  # cm
                "head_rot_w": quat_norm, "head_rot_x": 0.0, "head_rot_y": 0.0, "head_rot_z": 0.0,
                "right_hand_rot_w": 1.0, "left_hand_rot_w": 1.0,
                "take_id": take_id, "user_id": user_id, "game_id": game_id,
            })
            rows.append(row)
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False)


def test_quaternion_is_reordered_from_real_scalar_first_layout(tmp_path):
    """head_rot_w,x,y,z in the source -> UnitQuaternion.x,y,z,w in the output.
    A column-position-based read (rather than by name) would silently swap
    the scalar into the wrong slot without raising anything."""
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0)])
    frame = load_user_frame(tmp_path / "0.csv")
    _, _, session = next(split_sessions(frame))
    assert (session["UnitQuaternion.w"] == 1.0).all()
    assert (session["UnitQuaternion.x"] == 0.0).all()
    norm = np.linalg.norm(session[["UnitQuaternion.x", "UnitQuaternion.y",
                                    "UnitQuaternion.z", "UnitQuaternion.w"]].to_numpy(), axis=1)
    assert np.allclose(norm, 1.0)


def test_position_is_converted_from_centimetres_to_metres(tmp_path):
    """head_pos_y = 161.0 in the source (a real file's first row) must become
    1.61, not stay 161 -- the pipeline's other datasets are all in metres."""
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0)])
    frame = load_user_frame(tmp_path / "0.csv")
    _, _, session = next(split_sessions(frame))
    assert np.allclose(session["HmdPosition.y"], 1.61)


def test_timestamp_is_parsed_as_a_pandas_timedelta_and_zero_based(tmp_path):
    """The real column is a formatted duration string ("0 days
    00:18:47.969000"), not seconds and not epoch millis, and does not start
    at zero within a take."""
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0)], n=5, dt_ms=11)
    frame = load_user_frame(tmp_path / "0.csv")
    _, _, session = next(split_sessions(frame))
    assert session["SessionTime"].iloc[0] == 0.0
    assert np.allclose(session["SessionTime"].diff().dropna(), 0.011, atol=1e-6)


def test_game_and_take_each_produce_a_separate_session(tmp_path):
    """game_id carries the per-application label and must survive conversion
    as a distinct output file, not get merged or dropped -- it's the entire
    point of this dataset as a cross-application test set."""
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0), (3, 1), (4, 0)])
    frame = load_user_frame(tmp_path / "0.csv")
    sessions = list(split_sessions(frame))
    assert {(g, t) for g, t, _ in sessions} == {(3, 0), (3, 1), (4, 0)}


def test_only_numeric_stem_csvs_are_treated_as_participants(tmp_path):
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0)])
    (tmp_path / "notes.csv").write_text("not,a,participant\n")
    found = {user_id for user_id, _ in find_user_csvs(tmp_path)}
    assert found == {"0"}


def test_convert_writes_one_file_per_game_and_take_with_citation(tmp_path):
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0), (4, 0)])
    out = tmp_path / "out"
    rc = convert(tmp_path, out)
    assert rc == 0
    assert (out / "CITATION.txt").exists()
    assert (out / "0" / "beat_saber_take0.csv").exists()
    assert (out / "0" / "synth_riders_take0.csv").exists()


def test_convert_flags_but_still_writes_a_bad_quaternion_session(tmp_path):
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0)], quat_norm=5.0)
    out = tmp_path / "out"
    rc = convert(tmp_path, out)
    assert rc == 1
    assert (out / "0" / "beat_saber_take0.csv").exists()


def test_inspect_runs_without_writing_anything(tmp_path, capsys):
    _write_user_csv(tmp_path / "0.csv", 0, [(3, 0)])
    rc = inspect(tmp_path)
    assert rc == 0
    assert not (tmp_path / "out").exists()
    out = capsys.readouterr().out
    assert "23/9/17" in out, "the split-unknown warning must not be silently dropped"

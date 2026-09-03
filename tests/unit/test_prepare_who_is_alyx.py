"""
Tests for the who-is-alyx formatter, against the real source schema as reported from
a pulled copy: 41 columns, rotation in w,x,y,z order, position in centimetres, and
`delta_time_ms` holding milliseconds since session start.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prepare_who_is_alyx import (  # noqa: E402
    OUTPUT_COLUMNS,
    convert_session,
    describe,
    find_sessions,
)


pytestmark = pytest.mark.unit


def _source_csv(path: Path, rows: int = 200, hz: float = 20.0, start_ms: float = 0.0):
    """A file shaped like the real vr-controllers.csv, including the decoy columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time_ms = start_ms + np.arange(rows) * (1000.0 / hz)

    # Distinct per-axis values so a reordering bug cannot pass unnoticed.
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-17", periods=rows, freq="50ms").astype(str),
        "delta_time_ms": time_ms,
        "hmd_pos_x": np.full(rows, 150.0),     # centimetres
        "hmd_pos_y": np.full(rows, 170.0),
        "hmd_pos_z": np.full(rows, -25.0),
        "hmd_rot_w": np.full(rows, 1.0),
        "hmd_rot_x": np.zeros(rows),
        "hmd_rot_y": np.zeros(rows),
        "hmd_rot_z": np.zeros(rows),
        # Columns the pipeline must ignore.
        "left_controller_pos_x": np.full(rows, 999.0),
        "right_controller_trigger": np.zeros(rows),
        "left_controller_grip_button": np.zeros(rows),
    })
    frame.to_csv(path, index=False)
    return path


def test_converts_to_the_pipeline_schema(tmp_path):
    frame = convert_session(_source_csv(tmp_path / "vr-controllers.csv"), max_hz=None)
    assert list(frame.columns) == OUTPUT_COLUMNS


def test_centimetres_are_converted_to_metres(tmp_path):
    """Every other dataset is metres; a corpus that mixes units silently is a trap."""
    frame = convert_session(_source_csv(tmp_path / "vr-controllers.csv"), max_hz=None)
    assert frame["HmdPosition.x"].iloc[0] == pytest.approx(1.50)
    assert frame["HmdPosition.y"].iloc[0] == pytest.approx(1.70)
    assert frame["HmdPosition.z"].iloc[0] == pytest.approx(-0.25)


def test_quaternion_is_reordered_from_wxyz_to_xyzw(tmp_path):
    """
    The source order is w,x,y,z and the pipeline's is x,y,z,w. Getting this wrong
    yields a plausible-looking rotation that is silently mislabelled, so the identity
    quaternion here must land in .w and not in .x.
    """
    path = tmp_path / "vr-controllers.csv"
    _source_csv(path)
    frame = pd.read_csv(path)
    frame["hmd_rot_w"], frame["hmd_rot_x"] = 0.6, 0.8   # w != x, so order is testable
    frame["hmd_rot_y"], frame["hmd_rot_z"] = 0.0, 0.0
    frame.to_csv(path, index=False)

    converted = convert_session(path, max_hz=None)
    assert converted["UnitQuaternion.w"].iloc[0] == pytest.approx(0.6)
    assert converted["UnitQuaternion.x"].iloc[0] == pytest.approx(0.8)
    assert describe(converted)["quat_norm"] == pytest.approx(1.0, abs=1e-6)


def test_session_time_is_seconds_from_zero(tmp_path):
    """delta_time_ms is milliseconds since session start, and may not start at 0."""
    path = _source_csv(tmp_path / "vr-controllers.csv", rows=100, hz=20.0, start_ms=5000.0)
    frame = convert_session(path, max_hz=None)

    assert frame["SessionTime"].iloc[0] == pytest.approx(0.0)
    assert frame["SessionTime"].iloc[-1] == pytest.approx(99 * 0.05)
    assert describe(frame)["hz"] == pytest.approx(20.0, rel=0.02)


def test_decimation_caps_the_rate_and_is_off_when_below_it(tmp_path):
    fast = _source_csv(tmp_path / "fast" / "vr-controllers.csv", rows=2000, hz=100.0)
    capped = convert_session(fast, max_hz=50.0)
    assert describe(capped)["hz"] == pytest.approx(50.0, rel=0.1)

    slow = _source_csv(tmp_path / "slow" / "vr-controllers.csv", rows=400, hz=20.0)
    assert len(convert_session(slow, max_hz=50.0)) == 400, "no decimation below the cap"


def test_non_increasing_timestamps_are_dropped(tmp_path):
    """Sampler searches by time; a clock that goes backwards makes that meaningless."""
    path = tmp_path / "vr-controllers.csv"
    _source_csv(path, rows=50)
    frame = pd.read_csv(path)
    frame.loc[10, "delta_time_ms"] = frame.loc[5, "delta_time_ms"]   # a step backwards
    frame.to_csv(path, index=False)

    converted = convert_session(path, max_hz=None)
    assert converted["SessionTime"].is_monotonic_increasing
    assert len(converted) == 49


def test_non_finite_rows_are_dropped(tmp_path):
    path = tmp_path / "vr-controllers.csv"
    _source_csv(path, rows=50)
    frame = pd.read_csv(path)
    frame.loc[7, "hmd_pos_y"] = np.nan
    frame.to_csv(path, index=False)

    converted = convert_session(path, max_hz=None)
    assert len(converted) == 49
    assert np.isfinite(converted.to_numpy()).all()


def test_missing_required_column_is_an_error_not_a_guess(tmp_path):
    path = tmp_path / "vr-controllers.csv"
    _source_csv(path)
    frame = pd.read_csv(path).drop(columns=["hmd_rot_w"])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="hmd_rot_w"):
        convert_session(path, max_hz=None)


def test_finds_the_player_and_session_layout(tmp_path):
    """players/<player>/<session>/vr-controllers.csv, as reported from the real clone."""
    root = tmp_path / "who-is-alyx" / "players"
    for player, sessions in (("12", ["2022-01-17", "2022-01-24"]), ("74", ["2022-08-16"])):
        for session in sessions:
            _source_csv(root / player / session / "vr-controllers.csv", rows=20)
    # A session directory with no motion file must be skipped, not crash.
    (root / "12" / "2022-02-01").mkdir(parents=True)

    found = list(find_sessions(tmp_path / "who-is-alyx"))
    assert [(p, s) for p, s, _ in found] == [
        ("12", "2022-01-17"), ("12", "2022-01-24"), ("74", "2022-08-16"),
    ]


def test_controller_columns_are_not_carried_through(tmp_path):
    frame = convert_session(_source_csv(tmp_path / "vr-controllers.csv"), max_hz=None)
    assert not [c for c in frame.columns if "controller" in c]
    assert 999.0 not in frame.to_numpy(), "a controller value leaked into the output"

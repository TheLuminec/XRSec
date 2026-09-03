"""
Convert the who-is-alyx dataset into the pipeline's schema.

    python prepare_who_is_alyx.py --inspect --source /path/to/who-is-alyx
    python prepare_who_is_alyx.py --source /path/to/who-is-alyx

Source: https://github.com/cschell/who-is-alyx (Zenodo 10.5281/zenodo.8379914).
The CSVs are held in DVC, so the repo must be cloned and `dvc pull`ed first.

Why this dataset is worth the disk
----------------------------------
76 player directories, 146 sessions, and most players recorded TWO ~45-minute
sessions. That matters specifically because of two things measured in this project:

  - identity count is what limits the behavioural component. At 48 identities the
    movement-only arm sits at chance; at 343 it is clearly above. This adds ~76.
  - almost every player here has two sessions, so they can form genuine cross-session
    positive pairs. Only NJIT_6DOF in the existing corpus cannot, and cross-session
    pairing is now the required protocol.

What this script does to the data, and why
------------------------------------------
  columns    41 source columns -> the 8 the pipeline reads. Controller and button
             channels are dropped; the pipeline is head-motion only.
  time       SessionTime comes from `delta_time_ms` / 1000, which is milliseconds
             since session start. The `timestamp` column is absolute wall-clock
             ("YYYY-MM-DD HH:MM:SS.ffffff") and would need parsing for no gain.
  units      hmd_pos_* are centimetres; every other dataset in the corpus is metres,
             so they are divided by 100. `normalize=per_dataset` would absorb the
             scale difference anyway, but a raw corpus that silently mixes units is a
             trap for anyone reading it with normalize=none.
  quaternion the source order is w,x,y,z, NOT the x,y,z,w this pipeline uses.
             Columns are read BY NAME and written in pipeline order; the assertion
             below exists because getting this wrong produces a plausible-looking
             rotation that is silently mislabelled.
  rate       optionally decimated (see --max-hz). Late players record at ~98Hz while
             nothing in the pipeline samples above 20Hz, so the raw 6.7GB is mostly
             frames that get thrown away at load time.

Output: processed_datasets/who_is_alyx/users/<player>/<session>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MOTION_FILE = "vr-controllers.csv"

TIME_COLUMN = "delta_time_ms"
POSITION_COLUMNS = ["hmd_pos_x", "hmd_pos_y", "hmd_pos_z"]
# Source order is w,x,y,z. Read by name, never by position.
ROTATION_COLUMNS = ["hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w"]
REQUIRED = [TIME_COLUMN] + POSITION_COLUMNS + ROTATION_COLUMNS

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

CENTIMETRES_PER_METRE = 100.0


def find_sessions(source: Path):
    """Yield (player_id, session_id, csv_path) for every session with motion data."""
    players_root = source / "players" if (source / "players").is_dir() else source
    for player_dir in sorted(p for p in players_root.iterdir() if p.is_dir()):
        for session_dir in sorted(s for s in player_dir.iterdir() if s.is_dir()):
            motion = session_dir / MOTION_FILE
            if motion.is_file():
                yield player_dir.name, session_dir.name, motion


def convert_session(path: Path, max_hz: float | None) -> pd.DataFrame:
    """One source CSV -> a DataFrame in the pipeline's schema."""
    frame = pd.read_csv(path, usecols=lambda name: name in REQUIRED)

    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing {missing}")

    seconds = frame[TIME_COLUMN].to_numpy(dtype=float) / 1000.0
    position = frame[POSITION_COLUMNS].to_numpy(dtype=float) / CENTIMETRES_PER_METRE
    rotation = frame[ROTATION_COLUMNS].to_numpy(dtype=float)

    keep = np.isfinite(seconds) & np.isfinite(position).all(axis=1) & np.isfinite(rotation).all(axis=1)
    seconds, position, rotation = seconds[keep], position[keep], rotation[keep]

    # A non-increasing clock would make Sampler's nearest-point search meaningless.
    if seconds.size > 1:
        forward = np.concatenate([[True], np.diff(seconds) > 0])
        seconds, position, rotation = seconds[forward], position[forward], rotation[forward]

    if max_hz and seconds.size > 1:
        duration = seconds[-1] - seconds[0]
        native = seconds.size / duration if duration > 0 else 0.0
        if native > max_hz:
            step = max(1, int(round(native / max_hz)))
            seconds, position, rotation = seconds[::step], position[::step], rotation[::step]

    output = pd.DataFrame(
        np.column_stack([seconds, rotation, position]),
        columns=OUTPUT_COLUMNS,
    )
    output["SessionTime"] -= output["SessionTime"].iloc[0] if len(output) else 0.0
    return output


def describe(frame: pd.DataFrame) -> dict:
    if len(frame) < 2:
        return {"rows": len(frame), "hz": 0.0, "duration": 0.0, "quat_norm": float("nan")}
    duration = float(frame["SessionTime"].iloc[-1] - frame["SessionTime"].iloc[0])
    quaternion = frame[["UnitQuaternion.x", "UnitQuaternion.y",
                        "UnitQuaternion.z", "UnitQuaternion.w"]].to_numpy()
    return {
        "rows": len(frame),
        "duration": duration,
        "hz": len(frame) / duration if duration > 0 else 0.0,
        "quat_norm": float(np.linalg.norm(quaternion, axis=1).mean()),
    }


def inspect(source: Path, max_hz: float | None, limit: int = 4) -> int:
    sessions = list(find_sessions(source))
    players = sorted({player for player, _, _ in sessions})
    print(f"{len(players)} players, {len(sessions)} sessions with {MOTION_FILE}")

    per_player = {}
    for player, _, _ in sessions:
        per_player[player] = per_player.get(player, 0) + 1
    two_or_more = sum(1 for count in per_player.values() if count >= 2)
    print(f"players with >=2 sessions: {two_or_more} of {len(players)} "
          "(cross-session positives possible for these)")

    print(f"\nsampling {min(limit, len(sessions))} sessions:")
    for player, session, path in sessions[:: max(1, len(sessions) // max(limit, 1))][:limit]:
        stats = describe(convert_session(path, max_hz))
        print(f"  player {player:>3}  {session:<12} rows={stats['rows']:>7} "
              f"{stats['duration']:>8.1f}s  {stats['hz']:>6.1f}Hz  "
              f"|q|={stats['quat_norm']:.4f}")
    print("\n|q| must be ~1.0000. Anything else means the rotation columns are being "
          "read in the wrong order or are not a unit quaternion.")
    return 0


def convert(source: Path, out: Path, max_hz: float | None) -> int:
    sessions = list(find_sessions(source))
    if not sessions:
        print(f"ERROR: no {MOTION_FILE} found under {source}. Has `dvc pull` been run?")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    rows_total = 0
    bad_quaternions = []

    for player, session, path in sessions:
        try:
            frame = convert_session(path, max_hz)
        except Exception as exc:
            print(f"  SKIP player {player} {session}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        stats = describe(frame)
        if stats["rows"] < 2 or stats["duration"] <= 0:
            print(f"  SKIP player {player} {session}: unusable ({stats['rows']} rows, "
                  f"{stats['duration']:.1f}s)")
            skipped += 1
            continue
        if not (0.99 < stats["quat_norm"] < 1.01):
            bad_quaternions.append((player, session, stats["quat_norm"]))

        destination = out / player
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / f"{session}.csv", index=False)
        written += 1
        rows_total += stats["rows"]

    print(f"\nwrote {written} sessions ({rows_total:,} rows), skipped {skipped}")
    print(f"output: {out}")
    if bad_quaternions:
        print(f"\nWARNING: {len(bad_quaternions)} session(s) have a mean quaternion norm "
              "outside [0.99, 1.01]; the rotation columns may be misread:")
        for player, session, norm in bad_quaternions[:5]:
            print(f"  player {player} {session}: |q|={norm:.4f}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="Path to the who-is-alyx clone (after dvc pull)")
    parser.add_argument("--out", type=Path,
                        default=Path("processed_datasets/who_is_alyx/users"))
    parser.add_argument("--max-hz", type=float, default=60.0,
                        help="Decimate above this rate. Late players record ~98Hz and "
                             "nothing in the pipeline samples above 20Hz, so the extra "
                             "frames are discarded at load time anyway. 0 disables.")
    parser.add_argument("--inspect", action="store_true",
                        help="Report layout and per-session statistics without writing")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: {args.source} is not a directory")
        return 1

    max_hz = args.max_hz if args.max_hz and args.max_hz > 0 else None
    return inspect(args.source, max_hz) if args.inspect else convert(args.source, args.out, max_hz)


if __name__ == "__main__":
    sys.exit(main())

"""
Convert the "identification across XR applications" cross-app dataset
(Schach, Rack, McMahan, Latoschik -- Frontiers in Virtual Reality, 2026;
arXiv:2509.08539) into the pipeline's schema. HMD track only -- both
controller tracks are dropped, same convention as every other converter
here.

    python prepare_across_xr.py --inspect --source /path/to/csvs
    python prepare_across_xr.py --source /path/to/csvs

Source: University of Wurzburg GitLab,
gitlab2.informatik.uni-wuerzburg.de/hci/software/research-prototypes/
2025-frontiers-identification-across-xr-applications (README: Readme.md,
not README.md). One CSV per participant, 0.csv .. 48.csv, in-repo, ~5.4GB
total. Licence: CC BY-NC-SA 4.0, stated in the README (the GitLab API's
license/license_url fields report null for this project -- the licence
lives in the README text, not a LICENSE file, so checking the API alone
would wrongly conclude there is none).

DOWNLOAD ENDPOINT (per-file, no auth): GitLab's raw-file route --
  https://gitlab2.informatik.uni-wuerzburg.de/hci/software/
  research-prototypes/2025-frontiers-identification-across-xr-applications
  /-/raw/main/<N>.csv
for N in 0..48 -- confirmed working for /-/raw/main/Readme.md and (per a
peer session's Range-request mistake that pulled the full body instead of
a header slice, which at least confirms the file resolves) for /-/raw/main
/0.csv, ~109MB. AVALON's IP has been persistently 429'd on this host's
content-serving paths since first probing it -- 16+ hours as of this
writing, so treat it as a standing block on this machine rather than a
short-lived rate limit, and prefer fetching from elsewhere.

Note the trap that already cost time once: GitLab's API reports
license: null and license_url: null for this project, and separately, a
plain `git clone` over HTTPS returns a clean 403 for anonymous users on
this instance (deliberately disabled, not throttled) -- the raw-file route
above is the one that works anonymously.

THIS IS A TEST-ONLY DATASET. It exists to measure cross-application
transfer against the paper's own published numbers (78.5% averaged
accuracy, 83.1% within-application, both on their 17-user held-out test
split; see docs/LITERATURE_BRIEFING.md's "source X"), not to add training
identities. Convert it into its own directory, never pool it into a
training run, and see the PROVENANCE note this script writes for why.

Two conversion traps, both contradicted by what the paper text and the
README said, and both settled only by reading a real file's header --
worth remembering the next time a paper's prose is trusted over the data:

  - Quaternion order: the paper's text says "rotations (x, y, z, w)" and
    the README says "rotation (x, y, z)" (reads as Euler). The real CSV
    header has head_rot_w, head_rot_x, head_rot_y, head_rot_z -- scalar
    FIRST, w,x,y,z. That's the who-is-alyx trap, not what either source
    described. Columns are read by name and reordered to this pipeline's
    x,y,z,w; never assume position.
  - Units: head_pos_y on the first row of a real file is 161.00 -- not
    1.61. Centimetres, not the metres XROR/BOXRR used. Divided by 100.

Also real, also worth carrying forward:
  - timestamp is a pandas Timedelta string ("0 days 00:18:47.969000"), not
    seconds and not milliseconds. Parsed with pd.to_timedelta(...).
    total_seconds(). It does not start at zero within a take, so
    zero-basing per output session is still required.
  - Native rate is ~90Hz (rows ~11ms apart), not the 30fps quoted in the
    paper -- that 30fps is the paper's OWN preprocessing step, not a
    property of the released data. Their own config also applies a BRV
    (body-relative-velocity) encoding before training, which we do not
    reproduce here -- this script emits raw position + quaternion, this
    pipeline's standard schema. BRV needs a body frame built from head AND
    both controllers; on our own head-only extractor sweeps, encodings
    that drop absolute position lost by ~0.13 AUC to raw. Their using BRV
    successfully on a controller-equipped rig and raw winning on ours are
    the same fact from both sides, not a contradiction.
  - Coordinate system is right-x, up-y, forward-z -- same family as the
    rest of this corpus. No axis remap needed.

Session structure: game_id (1=Superhot VR, 2=Half-Life: Alyx, 3=Beat
Saber, 4=Synth Riders, 5=Social VR Scenario) and take_id (continuous
recording segments, separated by short in-sitting breaks -- NOT separate
days, unlike who-is-alyx's cross-session gap) together define a session.
One output CSV per (user_id, game_id, take_id) group, named so the
application is visible in the filename itself -- the per-application
label is the entire point of this dataset and must not get lost in
conversion the way it would if sessions were merged or renamed generically.

Split preservation: the paper trains on 23 users, validates on 9, tests
on the remaining 17 -- their headline 78.5%/83.1% numbers are measured on
exactly those 17. The split rule is read from their own
data_selection_slm.py (dataset-preprocessing repo, same GitLab group), not
guessed: participant CSVs sorted numerically by id, then

    train while user_id <  n_users * 0.45
    valid while user_id <  n_users * 0.65
    test  thereafter

No seed, no shuffle -- it is a pure id-ordered cut. For n_users=49 that is
train=0-22 (23 users), valid=23-31 (9 users), test=32-48 (17 users),
reproducing their reported 23/9/17 exactly -- a third independent
confirmation of this being the right dataset, after the 78.5%/83.1%/100%
number matches. `split_for_user()` implements this rule directly rather
than hard-coding the id list, so it stays correct if the dataset is ever
re-released with a different participant count. `convert()` writes the
assignment both as a `split` column in every session CSV and as a
top-level `splits.json` manifest, so "give me exactly their 17 test
users" doesn't require re-deriving the rule or scanning CSVs. Do not
re-split on our own choosing -- that is what turns this from a
measurement back into an argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CITATION_TEXT = """\
Data derived from the cross-application XR motion identification dataset:

  Schach, Lukas, Christian Rack, Ryan P. McMahan, and Marc Erich Latoschik.
  "Motion-Based User Identification across XR and Metaverse Applications
  by Deep Classification and Similarity Learning." Frontiers in Virtual
  Reality (2026). Also arXiv:2509.08539.

Licence: CC BY-NC-SA 4.0 (Attribution, NonCommercial, ShareAlike), as
stated in the source repository's README -- verify the exact citation
wording there before any public disclosure; this is the paper citation,
not necessarily the dataset's own preferred citation string.

TEST-ONLY. This dataset's value is a direct comparison against the
paper's own published numbers (78.5% averaged accuracy, 83.1%
within-application, both measured on their 17-user held-out test split).
It must never be pooled into a training run -- doing so would both
violate the point of the comparison and mix a NonCommercial-licensed
dataset's derived data into training artifacts more broadly than this
directory. Keep this directory, and anything derived from it, out of
`data_dirs` for any training config.
"""

RAW_TIME_COLUMN = "timestamp"
RAW_POSITION_COLUMNS = ["head_pos_x", "head_pos_y", "head_pos_z"]
# Real header order is scalar-first: w,x,y,z. Read by name, never by position.
RAW_ROTATION_COLUMNS = ["head_rot_x", "head_rot_y", "head_rot_z", "head_rot_w"]
ID_COLUMNS = ["take_id", "user_id", "game_id"]
REQUIRED = [RAW_TIME_COLUMN] + RAW_POSITION_COLUMNS + RAW_ROTATION_COLUMNS + ID_COLUMNS

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

CENTIMETRES_PER_METRE = 100.0

GAME_NAMES = {
    1: "superhot_vr",
    2: "half_life_alyx",
    3: "beat_saber",
    4: "synth_riders",
    5: "social_vr",
}

# From the source's own data_selection_slm.py (dataset-preprocessing repo):
# participant CSVs sorted numerically by id, no seed, no shuffle -- a pure
# id-ordered cut. Reproduces their reported 23/9/17 exactly at n_users=49.
TRAIN_FRACTION = 0.45
VALID_FRACTION = 0.20  # cumulative boundary is TRAIN_FRACTION + VALID_FRACTION


def split_for_user(user_id: int, n_users: int) -> str:
    """'train' / 'valid' / 'test', matching the paper's own split exactly."""
    if user_id < n_users * TRAIN_FRACTION:
        return "train"
    if user_id < n_users * (TRAIN_FRACTION + VALID_FRACTION):
        return "valid"
    return "test"


def find_user_csvs(source: Path):
    """Yield (user_id, csv_path) for every <n>.csv found, sorted numerically."""
    candidates = []
    for path in source.glob("*.csv"):
        if path.stem.isdigit():
            candidates.append((int(path.stem), path))
    for user_id, path in sorted(candidates):
        yield str(user_id), path


def load_user_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda name: name in REQUIRED)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return frame


def split_sessions(frame: pd.DataFrame):
    """Yield (game_id, take_id, session_frame) for each (user, game, take) group
    found in one participant's raw CSV, converted to the pipeline's schema."""
    seconds_all = pd.to_timedelta(frame[RAW_TIME_COLUMN]).dt.total_seconds().to_numpy()
    position_all = frame[RAW_POSITION_COLUMNS].to_numpy(dtype=float) / CENTIMETRES_PER_METRE
    rotation_all = frame[RAW_ROTATION_COLUMNS].to_numpy(dtype=float)
    game_ids = frame["game_id"].to_numpy()
    take_ids = frame["take_id"].to_numpy()

    for game_id in sorted(pd.unique(game_ids)):
        for take_id in sorted(pd.unique(take_ids[game_ids == game_id])):
            mask = (game_ids == game_id) & (take_ids == take_id)
            seconds = seconds_all[mask]
            position = position_all[mask]
            rotation = rotation_all[mask]

            keep = np.isfinite(seconds) & np.isfinite(position).all(axis=1) & np.isfinite(rotation).all(axis=1)
            seconds, position, rotation = seconds[keep], position[keep], rotation[keep]

            if seconds.size > 1:
                order = np.argsort(seconds, kind="stable")
                seconds, position, rotation = seconds[order], position[order], rotation[order]
                forward = np.concatenate([[True], np.diff(seconds) > 0])
                seconds, position, rotation = seconds[forward], position[forward], rotation[forward]

            output = pd.DataFrame(
                np.column_stack([seconds, rotation, position]),
                columns=OUTPUT_COLUMNS,
            )
            if len(output):
                output["SessionTime"] -= output["SessionTime"].iloc[0]
            yield int(game_id), int(take_id), output


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


def inspect(source: Path, limit: int = 4) -> int:
    users = list(find_user_csvs(source))
    print(f"{len(users)} participant CSVs found")
    if not users:
        return 0

    n_users = len(users)
    splits = [split_for_user(int(user_id), n_users) for user_id, _ in users]
    counts = {name: splits.count(name) for name in ("train", "valid", "test")}
    print(f"split (from data_selection_slm.py's own rule, n_users={n_users}): "
          f"train={counts['train']} valid={counts['valid']} test={counts['test']}")

    print(f"\nsampling {min(limit, len(users))} users:")
    shown = 0
    for user_id, path in users[:: max(1, len(users) // max(limit, 1))]:
        if shown >= limit:
            break
        raw = load_user_frame(path)
        sessions = list(split_sessions(raw))
        games = sorted({g for g, _, _ in sessions})
        stats = describe(sessions[0][2]) if sessions else describe(pd.DataFrame(columns=OUTPUT_COLUMNS))
        split = split_for_user(int(user_id), n_users)
        print(f"  user {user_id:>3} [{split:>5}]  {len(sessions)} (game,take) sessions, "
              f"games seen: {[GAME_NAMES.get(g, g) for g in games]}, "
              f"first session: rows={stats['rows']:>6} {stats['duration']:>6.1f}s "
              f"{stats['hz']:>5.1f}Hz |q|={stats['quat_norm']:.4f}")
        shown += 1
    print("\n|q| must be ~1.0000. Anything else means the rotation columns are being "
          "read in the wrong order or are not a unit quaternion.")
    return 0


def convert(source: Path, out: Path) -> int:
    users = list(find_user_csvs(source))
    if not users:
        print(f"ERROR: no <id>.csv files found under {source}")
        return 1

    n_users = len(users)
    split_by_user = {user_id: split_for_user(int(user_id), n_users) for user_id, _ in users}

    out.mkdir(parents=True, exist_ok=True)
    (out / "CITATION.txt").write_text(CITATION_TEXT)

    splits_manifest = {
        "rule": "user_id sorted numerically; train if id < n*0.45, "
                "valid if id < n*0.65, else test (from the source's own "
                "data_selection_slm.py -- no seed, no shuffle)",
        "n_users": n_users,
        "train": sorted((u for u, s in split_by_user.items() if s == "train"), key=int),
        "valid": sorted((u for u, s in split_by_user.items() if s == "valid"), key=int),
        "test": sorted((u for u, s in split_by_user.items() if s == "test"), key=int),
    }
    (out.parent / "splits.json").write_text(json.dumps(splits_manifest, indent=2))

    written = skipped = 0
    rows_total = 0
    bad_quaternions = []

    for user_id, path in users:
        try:
            raw = load_user_frame(path)
        except Exception as exc:
            print(f"  SKIP user {user_id}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        split = split_by_user[user_id]
        for game_id, take_id, frame in split_sessions(raw):
            stats = describe(frame)
            if stats["rows"] < 2 or stats["duration"] <= 0:
                print(f"  SKIP user {user_id} game={game_id} take={take_id}: "
                      f"unusable ({stats['rows']} rows, {stats['duration']:.1f}s)")
                skipped += 1
                continue
            if not (0.99 < stats["quat_norm"] < 1.01):
                bad_quaternions.append((user_id, game_id, take_id, stats["quat_norm"]))

            # 'split' rides along in the CSV too (harmless extra column --
            # the loader selects only its required columns by name) so the
            # assignment survives even if splits.json goes missing.
            frame = frame.copy()
            frame["split"] = split

            destination = out / user_id
            destination.mkdir(parents=True, exist_ok=True)
            game_name = GAME_NAMES.get(game_id, f"game{game_id}")
            frame.to_csv(destination / f"{game_name}_take{take_id}.csv", index=False)
            written += 1
            rows_total += stats["rows"]

    print(f"\nwrote {written} sessions ({rows_total:,} rows), skipped {skipped}")
    print(f"output: {out}")
    print(f"split manifest: {out.parent / 'splits.json'} "
          f"(train={len(splits_manifest['train'])} valid={len(splits_manifest['valid'])} "
          f"test={len(splits_manifest['test'])})")
    if bad_quaternions:
        print(f"\nWARNING: {len(bad_quaternions)} session(s) have a mean quaternion norm "
              "outside [0.99, 1.01]; the rotation columns may be misread:")
        for user_id, game_id, take_id, norm in bad_quaternions[:5]:
            print(f"  user {user_id} game={game_id} take={take_id}: |q|={norm:.4f}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="Directory containing the participant CSVs (0.csv .. 48.csv)")
    parser.add_argument("--out", type=Path,
                        default=Path("processed_datasets/CrossApplicationXR_Dataset/users"))
    parser.add_argument("--inspect", action="store_true",
                        help="Report layout and per-session statistics without writing")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: {args.source} is not a directory")
        return 1

    return inspect(args.source) if args.inspect else convert(args.source, args.out)


if __name__ == "__main__":
    sys.exit(main())

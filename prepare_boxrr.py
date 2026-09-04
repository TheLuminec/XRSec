"""
Convert BOXRR-23 (Berkeley Open Extended Reality Recordings 2023) into the
pipeline's schema. HMD track only -- both controller tracks are dropped.

    python prepare_boxrr.py --inspect --source /path/to/boxrr-23
    python prepare_boxrr.py --source /path/to/boxrr-23

Source: Nair, Guo, Mattern, Wang, O'Brien, Rosenberg, Song. "Berkeley Open
Extended Reality Recordings 2023 (BOXRR-23)". arXiv:2310.00430.
Distribution is gated behind an Ethical Dataset Use Agreement (a HIPAA-style
Limited Data Set agreement, not a click-through) at
rdi.berkeley.edu/metaverse/boxrr-23/dua.pdf -- clause 9 requires IRB (or
equivalent) approval of the research IN ADVANCE of use. That is an
institutional decision, made once, outside this script. This script assumes
it has already been settled by the time it is ever pointed at real data --
IT DOES NOT FETCH ANYTHING ITSELF.

Every citation-worthy public disclosure using this data must cite:
  Nair, Vivek, et al. "Unique Identification of 50,000+ Virtual Reality
  Users from Head & Hand Motion Data." arXiv:2302.08927 (2023).
`convert()` writes a CITATION.txt into the output directory carrying this,
so the requirement travels with the data rather than living in memory.

Format: XROR (github.com/MetaGuard/xror), BSON-encoded with an fpzip-
compressed frames array. Requires the `xror` package:
    pip install git+https://github.com/MetaGuard/xror.git#egg=xror
(pulls in fpzip, pymongo, bson -- none of which are in requirements.txt yet;
add them there once real conversion is actually happening, not before).

Device selection, worked out from xror/xror.py's own fromBSOR()/fromTilt()
converters rather than assumed
------------------------------------------------------------------------
Each XROR file declares its own device list under info.hardware.devices,
each entry a dict of {name, type, joint, axes}. `name` is an arbitrary
hardware string (e.g. a specific headset model) and is NOT a stable key --
BSOR-derived (Beat Saber) recordings tag the head device `type='HMD',
joint='HEAD'`, so this script selects on that, never on `name`.

Tilt Brush recordings converted by this library's own fromTilt() add only a
single 'BRUSH' device (type='OTHER') -- no HMD device at all. If that holds
for the released dataset too, the Tilt Brush half of BOXRR-23 may carry no
head track, and this script will report those recordings as skipped
("no HMD/HEAD device") rather than guess. Confirm this against real
--inspect output before assuming Tilt Brush is a lost cause.

Frame layout: `frames` is a 2D array, each row [time_seconds, then each
device's axes concatenated in device-list order]. Per-device axes for a
6DoF track are, by the library's own default and by fromBSOR's explicit
loop, ['x','y','z','i','j','k','1'] -- position xyz then quaternion with
the scalar LAST (i=qx, j=qy, k=qz, '1'=qw). That is already this pipeline's
x,y,z,w convention -- no reorder needed, unlike who-is-alyx's w,x,y,z. Still
read by declared axis name rather than assumed position, and the |q| check
below exists for the same reason it exists in prepare_who_is_alyx.py: a
wrong read produces a plausible-looking rotation, not a crash.

Units: XROR's coordinate convention is Unity's (left-handed, y-up), "1.0
units equals 1.0 meters" -- no scale conversion needed, unlike who-is-alyx's
centimetres.

Time: frame time is already seconds elapsed since the start of the
recording -- no ms conversion needed.

Sessions: BOXRR has ~45 recordings/user on average (Beat Saber rounds /
Tilt Brush sessions), each its own XROR file. Every file is treated as one
session, same as who-is-alyx's one-CSV-per-session convention -- do not
concatenate a user's recordings into one file, it would destroy
cross-session pairing.

Run this wherever the raw data actually lands
----------------------------------------------
Clause 4 of the DUA (no further distribution without Berkeley's written
consent) makes it unclear whether moving BOXRR-derived output from one
machine to another counts as distribution. Until that is settled, this
script is designed to be run in place -- point --source at wherever the raw
tarballs were extracted and --out at a local processed_datasets/ on that
same machine, rather than converting centrally and copying the result.

Termination note (DUA clause 15): on termination all copies must be
destroyed, including derived ones. That means processed_datasets/BOXRR-23/,
any checkpoints trained on it, AND its entries in .cache/samples/ (see
model/sample_cache.py) -- the cache is keyed off CSV names/sizes/mtimes and
will happily keep a serialized copy of converted BOXRR windows after the
source CSVs are gone. Whoever handles a termination needs to know to look
there too.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from xror import XROR
except ImportError:
    XROR = None

CITATION_TEXT = """\
Data derived from BOXRR-23 (Berkeley Open Extended Reality Recordings 2023).

Any public disclosure of research using this data must cite:
  Nair, Vivek, Wenbo Guo, Justus Mattern, Rui Wang, James F. O'Brien,
  Louis Rosenberg, and Dawn Song. "Unique Identification of 50,000+
  Virtual Reality Users from Head & Hand Motion Data." arXiv, 17 February
  2023. doi:10.48550/arXiv.2302.08927.

Governed by an Ethical Dataset Use Agreement with The Regents of the
University of California (rdi.berkeley.edu/metaverse/boxrr-23/dua.pdf).
Notably: no further distribution without UC Berkeley's written consent
(clause 4); on termination, all copies -- including this directory, any
checkpoints trained on it, and any .cache/samples/ entries derived from
it -- must be destroyed (clause 15).
"""

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

# Axis-name -> output-column mapping for one 6DoF device, per the XROR
# convention (position xyz, quaternion with scalar last as i,j,k,'1').
POSITION_AXES = {"x": "HmdPosition.x", "y": "HmdPosition.y", "z": "HmdPosition.z"}
ROTATION_AXES = {
    "i": "UnitQuaternion.x", "j": "UnitQuaternion.y",
    "k": "UnitQuaternion.z", "1": "UnitQuaternion.w",
}


def _require_xror():
    if XROR is None:
        raise ImportError(
            "the `xror` package is not installed. "
            "pip install git+https://github.com/MetaGuard/xror.git#egg=xror"
        )


def find_sessions(source: Path, max_per_user: int | None = None):
    """Yield (user_id, recording_id, xror_path) for every .xror file found.

    Accepts either an already-extracted tree (source/users/<id>/*.xror or
    source/<id>/*.xror) or per-user .tar archives (source/users/<id>.tar),
    matching the "one tarball per user" layout HuggingFace describes.
    Archives are NOT extracted automatically -- see extract_user_tarballs().

    max_per_user caps recordings kept per user (sorted by recording id, so
    the choice is deterministic across runs). It does not reduce what gets
    downloaded -- a user's tarball always contains every one of their
    recordings -- only what gets converted. Select users with few enough
    recordings in the first place (see the metadata BSON index) to avoid
    downloading recordings that will just be discarded here.
    """
    root = source / "users" if (source / "users").is_dir() else source
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            user_id = entry.name
            xror_paths = sorted(entry.glob("*.xror"))
            if max_per_user is not None:
                xror_paths = xror_paths[:max_per_user]
            for xror_path in xror_paths:
                yield user_id, xror_path.stem, xror_path
        elif entry.suffix == ".tar":
            # Not extracted yet -- report via find_unextracted_tarballs(),
            # not silently skipped.
            continue


def find_unextracted_tarballs(source: Path):
    root = source / "users" if (source / "users").is_dir() else source
    if not root.is_dir():
        return []
    return sorted(root.glob("*.tar"))


def extract_user_tarball(tar_path: Path, dest_root: Path) -> Path:
    """Extract one user's tarball (tar xvf <id>.tar) into dest_root/.

    The tarball's own top-level entry IS the user_id directory (confirmed
    against real files: `tar tf <id>.tar` lists `<id>/`, `<id>/<replay>.xror`,
    ...) -- extracting into dest_root/<id>/ as a first cut double-nested it
    into dest_root/<id>/<id>/*.xror. Extract straight into dest_root instead.
    """
    user_id = tar_path.stem
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_root, filter="data")
    return dest_root / user_id


def _device_axis_offsets(devices: list[dict]) -> list[tuple[dict, int]]:
    """(device, start_column) for each device, columns counted after time."""
    offsets = []
    column = 0
    for device in devices:
        offsets.append((device, column))
        column += len(device.get("axes", []))
    return offsets


def find_hmd_device(devices: list[dict]):
    """Locate the head device by type/joint, never by the free-text name."""
    for device, offset in _device_axis_offsets(devices):
        if device.get("type") == "HMD" or device.get("joint") == "HEAD":
            return device, offset
    return None, None


def convert_session(path: Path) -> pd.DataFrame:
    """One .xror file -> a DataFrame in the pipeline's schema, HMD only."""
    _require_xror()
    with open(path, "rb") as f:
        raw = f.read()
    xror = XROR.unpack(raw)

    devices = xror.data.get("info", {}).get("hardware", {}).get("devices", [])
    device, offset = find_hmd_device(devices)
    if device is None:
        names = [d.get("name", "?") for d in devices]
        types = [d.get("type", "?") for d in devices]
        raise ValueError(f"no HMD/HEAD device in {path.name} (devices: "
                         f"{list(zip(names, types))})")

    axes = device.get("axes", [])
    column_for = {}
    for i, axis in enumerate(axes):
        out_col = POSITION_AXES.get(axis) or ROTATION_AXES.get(axis)
        if out_col:
            column_for[out_col] = offset + i
    missing = [c for c in OUTPUT_COLUMNS[1:] if c not in column_for]
    if missing:
        raise ValueError(f"{path.name} HMD device is missing axes for {missing} "
                         f"(declared axes: {axes})")

    frames = np.asarray(xror.data.get("frames", []), dtype=float)
    if frames.size == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Column offsets come from the DECLARED device list, so if the frame width
    # disagrees with that declaration the offsets are wrong and every column read is
    # off by some amount - which means reading a controller's channels as the head's.
    # Nothing downstream would catch it: a controller quaternion is also unit-norm, and
    # hand motion correlates with head motion enough to look like plausible data. Fail
    # instead of guessing.
    expected = 1 + sum(len(d.get("axes", [])) for d in devices)
    if frames.ndim != 2 or frames.shape[1] != expected:
        raise ValueError(
            f"{path.name}: frame width {frames.shape[1] if frames.ndim == 2 else '?'} "
            f"does not match the {expected} columns declared by "
            f"{[d.get('type', '?') for d in devices]}. Column offsets would be wrong, "
            f"so the HMD track cannot be located safely.")

    seconds = frames[:, 0]
    columns = {"SessionTime": seconds}
    for out_col in OUTPUT_COLUMNS[1:]:
        columns[out_col] = frames[:, 1 + column_for[out_col]]
    frame = pd.DataFrame(columns)

    finite = np.isfinite(frame.to_numpy()).all(axis=1)
    frame = frame[finite]

    if len(frame) > 1:
        forward = np.concatenate([[True], np.diff(frame["SessionTime"].to_numpy()) > 0])
        frame = frame[forward]

    frame = frame.reset_index(drop=True)
    if len(frame):
        frame["SessionTime"] -= frame["SessionTime"].iloc[0]
    return frame[OUTPUT_COLUMNS]


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
    tarballs = find_unextracted_tarballs(source)
    if tarballs:
        print(f"{len(tarballs)} user tarball(s) not yet extracted "
              f"(e.g. {tarballs[0].name}) -- extract with "
              "extract_user_tarball() or `tar xvf` before --inspect can see them.")

    sessions = list(find_sessions(source))
    users = sorted({user for user, _, _ in sessions})
    print(f"{len(users)} users, {len(sessions)} recordings with .xror files")
    if not sessions:
        return 0

    per_user = {}
    for user, _, _ in sessions:
        per_user[user] = per_user.get(user, 0) + 1
    two_or_more = sum(1 for count in per_user.values() if count >= 2)
    print(f"users with >=2 recordings: {two_or_more} of {len(users)} "
          "(cross-session positives possible for these)")

    print(f"\nsampling {min(limit, len(sessions))} recordings:")
    skipped_no_hmd = 0
    shown = 0
    step = max(1, len(sessions) // max(limit * 4, 1))
    for user, recording, path in sessions[::step]:
        if shown >= limit:
            break
        try:
            stats = describe(convert_session(path))
        except ValueError as exc:
            print(f"  SKIP user {user} {recording}: {exc}")
            skipped_no_hmd += 1
            continue
        print(f"  user {user:>8}  {recording:<20} rows={stats['rows']:>7} "
              f"{stats['duration']:>8.1f}s  {stats['hz']:>6.1f}Hz  "
              f"|q|={stats['quat_norm']:.4f}")
        shown += 1
    if skipped_no_hmd:
        print(f"\n{skipped_no_hmd} sampled recording(s) had no HMD/HEAD device "
              "(check whether these are Tilt Brush -- see module docstring).")
    print("\n|q| must be ~1.0000. Anything else means the rotation axes are being "
          "read in the wrong order or are not a unit quaternion.")
    return 0


def convert(source: Path, out: Path, max_per_user: int | None = None) -> int:
    sessions = list(find_sessions(source, max_per_user=max_per_user))
    if not sessions:
        tarballs = find_unextracted_tarballs(source)
        if tarballs:
            print(f"ERROR: {len(tarballs)} user tarball(s) found but not extracted. "
                  "Extract them first (tar xvf <id>.tar into users/<id>/).")
        else:
            print(f"ERROR: no .xror files found under {source}.")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "CITATION.txt").write_text(CITATION_TEXT)

    written = skipped = 0
    rows_total = 0
    bad_quaternions = []

    for user, recording, path in sessions:
        try:
            frame = convert_session(path)
        except Exception as exc:
            print(f"  SKIP user {user} {recording}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        stats = describe(frame)
        if stats["rows"] < 2 or stats["duration"] <= 0:
            print(f"  SKIP user {user} {recording}: unusable ({stats['rows']} rows, "
                  f"{stats['duration']:.1f}s)")
            skipped += 1
            continue
        if not (0.99 < stats["quat_norm"] < 1.01):
            bad_quaternions.append((user, recording, stats["quat_norm"]))

        destination = out / user
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / f"{recording}.csv", index=False)
        written += 1
        rows_total += stats["rows"]

    print(f"\nwrote {written} sessions ({rows_total:,} rows), skipped {skipped}")
    print(f"output: {out}")
    if bad_quaternions:
        print(f"\nWARNING: {len(bad_quaternions)} session(s) have a mean quaternion norm "
              "outside [0.99, 1.01]; the rotation axes may be misread:")
        for user, recording, norm in bad_quaternions[:5]:
            print(f"  user {user} {recording}: |q|={norm:.4f}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="Path to extracted BOXRR-23 users/ directory")
    parser.add_argument("--out", type=Path,
                        default=Path("processed_datasets/BOXRR-23_Dataset/users"))
    parser.add_argument("--inspect", action="store_true",
                        help="Report layout and per-recording statistics without writing")
    parser.add_argument("--max-per-user", type=int, default=None,
                        help="Cap recordings converted per user (sorted by recording id). "
                             "Does not reduce what was downloaded, only what gets written.")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: {args.source} is not a directory")
        return 1

    return inspect(args.source) if args.inspect else convert(args.source, args.out, args.max_per_user)


if __name__ == "__main__":
    sys.exit(main())

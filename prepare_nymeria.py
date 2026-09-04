"""
Convert Nymeria (Meta / Project Aria glasses) into the pipeline's schema.
Head-glasses track only -- wrist recordings, the Xsens body suit, RGB/eye-
tracking video, and the third-person observer recording are all dropped.

    python prepare_nymeria.py --fetch --url-json nymeria_download_urls.json \
        --keys-file keys.txt --dest raw/
    python prepare_nymeria.py --inspect --source raw/
    python prepare_nymeria.py --source raw/

Source: Ma et al., "Nymeria: A Massive Collection of Multimodal Egocentric
Daily Motion in the Wild", ECCV 2024, arXiv:2406.09905.
huggingface.co/datasets/projectaria/Nymeria (code + two small metadata
files, not gated) · projectaria.com/datasets/nymeria (the actual sequence
data, gated behind an interactive login + licence-acceptance flow on
explorer.projectaria.com that produces a signed, per-sequence URL list --
accepting that flow is the user's action, not this script's or any
session's; this script only ever consumes the resulting JSON). Licence:
CC BY-NC 4.0.

Why this dataset is worth the disk despite everything else in the corpus
already being head-only: it is the only dataset found that is actually
captured on real XR glasses (Project Aria) rather than a VR headset
standing in for one, and it is the most task-diverse capture in the
corpus -- unscripted daily activities across 50 real-world locations.

Selective download, because the alternative is ~80TB
------------------------------------------------------
Nymeria's `download.py` downloads whole DataGroups, one zip per group per
sequence -- there is no per-file selection within a group, and Range
requests are NOT honoured by the CDN serving the signed URLs (confirmed:
a `curl -r 0-1023` against a real download_url returns a plain 200, not
206 Partial Content), so there is no remote-zip trick for pulling just
one member without downloading the whole zip. The `recording_head` group
bundles `data/motion.vrs`, `data/et.vrs` (eye-tracking video),
`mps/slam/closed_loop_trajectory.csv` (what we want),
`mps/slam/semidense_points.csv.gz`, `mps/slam/online_calibration.jsonl`,
`mps/slam/summary.json`, and `mps/eye_gaze/*` -- averaging ~689MB/sequence
across a 1,100-sequence sample (measured from the real url-list JSON's own
file_size_bytes fields, min 361MB, max 1964MB). Measured overhead from 3
real zips: closed_loop_trajectory.csv is itself the LARGEST member (~370-
390MB uncompressed, because Aria's closed-loop trajectory runs at ~1kHz
for a ~20-minute sequence) -- so the "waste" of pulling the whole zip for
one file is roughly 30%, not the 99% a video-dominated bundle might
suggest. `fetch_sequences()` downloads the zip to a temp path, extracts
ONLY closed_loop_trajectory.csv via zipfile (no full extraction to disk),
and deletes the zip -- this halves nothing about the download itself but
avoids leaving ~300MB/sequence of video and point-cloud data on disk
afterward.

Selecting WHICH sequences is a policy decision (how many identities,
how many activities each, from which url-list) made by the caller, not
this script -- see the project chat log around 2026-09-04 for the sizing
argument (roughly 100 participants x 3 sequences lands in the tens-of-GB
range at ~689MB/sequence for the head zip alone, before the ~300MB/
sequence of discarded video is subtracted back out).

FORMAT -- verified against 3 real closed_loop_trajectory.csv files, not
assumed from documentation. This is the cleanest converter in the corpus:
NEITHER conversion trap that has bitten every other dataset applies here.

  - Quaternion is ALREADY this pipeline's x,y,z,w order: columns are named
    qx_world_device, qy_world_device, qz_world_device, qw_world_device --
    scalar last, unlike who-is-alyx's and the cross-app dataset's scalar-
    first layout. Still read and renamed by column NAME, not position,
    same discipline as everywhere else -- measured mean |q| = 0.99999999997
    over 1,000 real rows confirms no reorder is needed.
  - Position is ALREADY metres: tx/ty/tz_world_device, small values near
    the trajectory's start (world-frame SLAM trajectories begin close to
    the origin and grow from there -- consistent with this sequence's
    reported 234m total trajectory_length_m).
  - Timestamp is tracking_timestamp_us, an integer count of MICROSECONDS
    on the device's own monotonic clock (not the coarser-grained,
    UTC-epoch utc_timestamp_ns column, which updates far less often --
    584 distinct values across 20,001 rows in the file checked -- and is
    dropped). Divided by 1e6 for SessionTime, zero-based per session as
    everywhere else.
  - Native rate is ~1029Hz (measured: mean inter-row delta 971.9us over
    20,000 rows), confirmed monotonic. Comfortably above the 20Hz floor.

The world_device pose is the GLASSES frame, not head centre -- a fixed
per-wearer offset (glasses sit in roughly the same place on everyone's
face relative to their skull), so it is consistent within a user and
harmless for identification, but it is a different reference point than
a VR headset's tracked origin and should not be assumed identical without
checking if it ever matters for a cross-dataset comparison.

Every recording_head file also carries a second, ECEF-frame copy of the
same pose (tx/ty/tz_ecef_device, qx/qy/qz/qw_ecef_device, gated on
geo_available) -- an absolute geo-referenced pose. Not used; world_device
is this pipeline's frame of reference for every other dataset too.

Sessions and the cross-activity-not-cross-day caveat
------------------------------------------------------
Sequence keys are `<date>_<session>_<fake_name>_act<N>_<uid>`, e.g.
`20230607_s0_james_johnson_act0_e72nhq`. `fake_name` is the identity key
(the participants_metadata.csv header says plainly: "All names are fake.
They are not related to participants." -- a privacy label, not a data
defect) and is 1:1 with one participant's one sitting: every participant
appears on exactly one date, in exactly one session, wearing the glasses
throughout. Each actN is a different UNSCRIPTED ACTIVITY within that same
sitting, not a separate day. Treat each actN as a session for pairing
purposes -- the glasses come off between nothing, so this is genuinely
independent motion data per activity -- but this dataset contributes
CROSS-ACTIVITY positives, not cross-day ones, and must never be quoted as
evidence of temporal persistence the way who-is-alyx's two-different-days
structure can be.

The most valuable file in the download
------------------------------------------------------
`data/participants_metadata.csv` (fetched separately, not gated, ~31KB)
carries MEASURED height_cm (149-199, mean 169.7 across 275 of 287 rows
with a value) plus shoulder/hip/knee height, arm span, weight and BMI for
every participant -- including the 236 with released sequences. This
pipeline's central finding, that roughly 78% of what the model uses is
absolute head position (i.e. height and posture), has only ever been
INFERRED by centring the position channel and watching accuracy collapse.
With 236 people whose height is directly measured rather than inferred
from motion, that becomes a testable claim rather than an inference for
the first time. `convert()` copies this file into the output directory
unmodified -- do not drop it as irrelevant metadata.

CITATION.txt is written into the output directory, same as every other
converter here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

CITATION_TEXT = """\
Data derived from the Nymeria dataset:

  Ma, Lingni, Yuting Ye, Fangzhou Hong, Vladimir Guzov, Yifeng Jiang,
  Rowan Postyeni, Luis Pesqueira, Alexander Gamino, Vijay Baiyya, Hyo Jin
  Kim, Kevin Bailey, David Soriano Fosas, C. Karen Liu, Ziwei Liu, Jakob
  Engel, Renzo De Nardi, and Richard Newcombe. "Nymeria: A Massive
  Collection of Multimodal Egocentric Daily Motion in the Wild."
  Proceedings of the 18th European Conference on Computer Vision (ECCV),
  2024. arXiv:2406.09905.

Licence: CC BY-NC 4.0 (Attribution, NonCommercial). Data and code may not
be used for commercial purposes.

CROSS-ACTIVITY, NOT CROSS-DAY. Every participant appears in exactly one
date and one session; the multiple recordings per participant (actN) are
different unscripted activities within a single sitting, glasses mounted
throughout, not separate days. Treat as sessions for pairing purposes but
do not cite as evidence of identification persisting over time -- see
who-is-alyx or the Stanford Longitudinal corpus (if ever acquired) for
genuine cross-day structure.

The recorded pose is the GLASSES frame, not head centre -- a fixed
per-wearer offset, consistent within a user, but a different reference
point than a VR headset's tracked origin.

participants_metadata.csv (carried in this directory unmodified) has
MEASURED height/weight/BMI/limb-length data for every participant --
worth using directly rather than only as an identity join key.
"""

PARTICIPANTS_METADATA_FILENAME = "participants_metadata.csv"

SEQUENCE_KEY_RE = re.compile(r"^(\d{8})_(s\d+)_(.+)_act(\d+)_([a-z0-9]+)$")

RAW_TIME_COLUMN = "tracking_timestamp_us"
RAW_POSITION_COLUMNS = ["tx_world_device", "ty_world_device", "tz_world_device"]
RAW_ROTATION_COLUMNS = ["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
REQUIRED = [RAW_TIME_COLUMN] + RAW_POSITION_COLUMNS + RAW_ROTATION_COLUMNS

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

MICROSECONDS_PER_SECOND = 1_000_000.0

TRAJECTORY_MEMBER = "recording_head/mps/slam/closed_loop_trajectory.csv"


def parse_sequence_key(key: str):
    """(date, session, fake_name, act_id, uid) or None if it doesn't match."""
    m = SEQUENCE_KEY_RE.match(key)
    if not m:
        return None
    date, session, fake_name, act_id, uid = m.groups()
    return date, session, fake_name, act_id, uid


def fetch_sequences(url_json: Path, sequence_keys, dest: Path, group: str = "recording_head") -> list[Path]:
    """Download `group`'s zip for each sequence key, extract ONLY the
    trajectory CSV member, delete the zip. Returns the extracted CSV paths.

    Network access lives here and only here -- convert()/inspect() never
    touch it, so they can be tested without downloading anything.
    """
    with open(url_json) as f:
        manifest = json.load(f)
    sequences = manifest["sequences"]

    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for key in sequence_keys:
        entry = sequences.get(key, {}).get(group)
        if entry is None:
            print(f"  SKIP {key}: not present in url-list for group {group!r}")
            continue

        target = dest / key / "closed_loop_trajectory.csv"
        if target.exists():
            written.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            print(f"  fetching {key} ({entry['file_size_bytes'] / 1e6:.0f} MB zip)...")
            with urlopen(entry["download_url"]) as response:
                while chunk := response.read(1 << 20):
                    tmp.write(chunk)
            tmp.flush()
            with zipfile.ZipFile(tmp.name) as zf:
                with zf.open(TRAJECTORY_MEMBER) as member, open(target, "wb") as out:
                    out.write(member.read())
        written.append(target)
    return written


def find_sequences(source: Path):
    """Yield (fake_name, act_id, csv_path) for every already-fetched
    closed_loop_trajectory.csv under source/<sequence_key>/."""
    for entry in sorted(source.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_sequence_key(entry.name)
        csv_path = entry / "closed_loop_trajectory.csv"
        if parsed and csv_path.is_file():
            _, _, fake_name, act_id, _ = parsed
            yield fake_name, act_id, csv_path


def convert_session(path: Path) -> pd.DataFrame:
    """One closed_loop_trajectory.csv -> a DataFrame in the pipeline's schema."""
    frame = pd.read_csv(path, usecols=lambda name: name in REQUIRED)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing {missing}")

    seconds = frame[RAW_TIME_COLUMN].to_numpy(dtype=float) / MICROSECONDS_PER_SECOND
    position = frame[RAW_POSITION_COLUMNS].to_numpy(dtype=float)
    rotation = frame[RAW_ROTATION_COLUMNS].to_numpy(dtype=float)

    keep = np.isfinite(seconds) & np.isfinite(position).all(axis=1) & np.isfinite(rotation).all(axis=1)
    seconds, position, rotation = seconds[keep], position[keep], rotation[keep]

    if seconds.size > 1:
        forward = np.concatenate([[True], np.diff(seconds) > 0])
        seconds, position, rotation = seconds[forward], position[forward], rotation[forward]

    output = pd.DataFrame(
        np.column_stack([seconds, rotation, position]),
        columns=OUTPUT_COLUMNS,
    )
    if len(output):
        output["SessionTime"] -= output["SessionTime"].iloc[0]
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


def inspect(source: Path, limit: int = 4) -> int:
    sequences = list(find_sequences(source))
    users = sorted({name for name, _, _ in sequences})
    print(f"{len(users)} participants, {len(sequences)} activity recordings found")
    if not sequences:
        return 0

    per_user = {}
    for name, _, _ in sequences:
        per_user[name] = per_user.get(name, 0) + 1
    two_or_more = sum(1 for count in per_user.values() if count >= 2)
    print(f"participants with >=2 activities: {two_or_more} of {len(users)} "
          "(cross-activity positives possible for these -- see docstring: "
          "cross-activity, not cross-day)")

    print(f"\nsampling {min(limit, len(sequences))} recordings:")
    for name, act_id, path in sequences[:: max(1, len(sequences) // max(limit, 1))][:limit]:
        stats = describe(convert_session(path))
        print(f"  {name:<20} act{act_id}  rows={stats['rows']:>8} "
              f"{stats['duration']:>8.1f}s  {stats['hz']:>7.1f}Hz  "
              f"|q|={stats['quat_norm']:.6f}")
    print("\n|q| must be ~1.000000. Anything else means the rotation columns are "
          "being read in the wrong order or are not a unit quaternion.")
    return 0


def convert(source: Path, out: Path, metadata_csv: Path | None = None) -> int:
    sequences = list(find_sequences(source))
    if not sequences:
        print(f"ERROR: no closed_loop_trajectory.csv found under {source}")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "CITATION.txt").write_text(CITATION_TEXT)
    if metadata_csv and metadata_csv.is_file():
        (out / PARTICIPANTS_METADATA_FILENAME).write_bytes(metadata_csv.read_bytes())
    elif metadata_csv:
        print(f"  WARNING: --participants-metadata {metadata_csv} not found; "
              "converting without it (see module docstring for why it matters)")

    written = skipped = 0
    rows_total = 0
    bad_quaternions = []

    for fake_name, act_id, path in sequences:
        try:
            frame = convert_session(path)
        except Exception as exc:
            print(f"  SKIP {fake_name} act{act_id}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        stats = describe(frame)
        if stats["rows"] < 2 or stats["duration"] <= 0:
            print(f"  SKIP {fake_name} act{act_id}: unusable ({stats['rows']} rows, "
                  f"{stats['duration']:.1f}s)")
            skipped += 1
            continue
        if not (0.99 < stats["quat_norm"] < 1.01):
            bad_quaternions.append((fake_name, act_id, stats["quat_norm"]))

        destination = out / fake_name
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / f"act{act_id}.csv", index=False)
        written += 1
        rows_total += stats["rows"]

    print(f"\nwrote {written} sessions ({rows_total:,} rows), skipped {skipped}")
    print(f"output: {out}")
    if bad_quaternions:
        print(f"\nWARNING: {len(bad_quaternions)} session(s) have a mean quaternion norm "
              "outside [0.99, 1.01]; the rotation columns may be misread:")
        for fake_name, act_id, norm in bad_quaternions[:5]:
            print(f"  {fake_name} act{act_id}: |q|={norm:.4f}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path,
                        help="Directory of already-fetched <sequence_key>/closed_loop_trajectory.csv")
    parser.add_argument("--out", type=Path,
                        default=Path("processed_datasets/Nymeria_Dataset/users"))
    parser.add_argument("--participants-metadata", type=Path, default=None,
                        help="Path to participants_metadata.csv, copied into the output "
                             "directory unmodified (see module docstring)")
    parser.add_argument("--inspect", action="store_true",
                        help="Report layout and per-recording statistics without writing")
    parser.add_argument("--fetch", action="store_true",
                        help="Download trajectory CSVs for --keys-file's sequence keys "
                             "using --url-json, into --dest, then exit")
    parser.add_argument("--url-json", type=Path, default=None,
                        help="nymeria_download_urls.json (obtained via the user's own "
                             "login/licence-acceptance at explorer.projectaria.com)")
    parser.add_argument("--keys-file", type=Path, default=None,
                        help="One sequence key per line")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Where --fetch writes <sequence_key>/closed_loop_trajectory.csv")
    args = parser.parse_args()

    if args.fetch:
        if not (args.url_json and args.keys_file and args.dest):
            print("ERROR: --fetch requires --url-json, --keys-file and --dest")
            return 1
        keys = [line.strip() for line in args.keys_file.read_text().splitlines() if line.strip()]
        written = fetch_sequences(args.url_json, keys, args.dest)
        print(f"fetched {len(written)} of {len(keys)} requested sequences")
        return 0

    if not args.source or not args.source.is_dir():
        print(f"ERROR: --source {args.source} is not a directory")
        return 1

    return (inspect(args.source) if args.inspect
            else convert(args.source, args.out, args.participants_metadata))


if __name__ == "__main__":
    sys.exit(main())

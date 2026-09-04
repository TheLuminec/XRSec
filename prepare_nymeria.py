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

UP AXIS IS Z, NOT Y IN THE RAW DATA -- verified, not assumed, and this
was the one real axis-convention break in the corpus (every other
dataset here -- BOXRR-23/who-is-alyx/across-xr, all Unity-family -- is
left-handed Y-up). Nymeria's raw `gravity_z_world` column reads a
constant -9.81 with gravity_x_world/gravity_y_world at 0.0 across every
row checked -- gravity points along -Z, so Z is vertical.

TWO SEPARATE FIXES ARE NEEDED, NOT ONE, and finding this out cost a
retraction (see git history around 2026-09-04). `up_axis="z"` (the
default) applies BOTH, in order:

  1. rotate_z_up_to_y_up() -- a WORLD-side fix: rotates the -90deg-about-X
     change of basis into both position and orientation so height lands
     in HmdPosition.y and gravity transforms to (0,-9.81,0) exactly. This
     alone was shipped once and looked complete (gravity check passed,
     mean |q| stayed 1.0) but was wrong in a way neither of those checks
     could catch: it fixes which WORLD frame the pose is expressed in,
     but leaves the quaternion still expressing rotation FROM Aria's own
     device axes (the local frame of the left SLAM camera, physically
     mounted at an angle on the glasses temple -- NOT the wearer's head-
     forward direction) rather than a Unity head frame. Rotating local
     +Y (nominally "up") by the once-fixed quaternion gave a near-
     meaningless 0.15-magnitude result, not world up.
  2. fix_device_frame() -- the DEVICE-side fix this was missing: remaps
     FROM this pipeline's Unity head-frame convention (x=right,y=up,
     z=forward) TO Aria's native device axes, derived from real
     calibration data (T_Device_Camera for the forward-facing RGB
     camera) rather than guessed or taken from generic per-camera-frame
     documentation, which does not directly apply here -- see
     fix_device_frame()'s own docstring for the full derivation,
     including a determinant check that caught a reflection-vs-rotation
     sign error while building it.

Both are needed together for `HmdPosition.y`/local device "up" to mean
what they mean everywhere else in the corpus. Verified on ALL 100 real,
already-converted sequences (not a handful): pooled local +Y -> world up
concentration 0.9127, mean |q| = 1.0000000000, and a locomotion-only
forward-direction check (median cosine alignment 0.86, 80% of moving
windows positive) -- see fix_device_frame()'s docstring and
PROVENANCE.md for the full per-script table and the test that separates
"wrong constant" from "real head-tilting during the activity" for the
two scripts that dipped below the check's diagnostic threshold.

PRACTICAL CONSEQUENCE: with `up_axis="z"` (default), HEIGHT LIVES IN
HmdPosition.y like every other dataset in the corpus -- no special-casing
needed for a height-vs-participants_metadata.csv ground-truth comparison
or anything else that assumes column identity implies physical axis
identity. Pass `up_axis="y"` only to inspect the raw, unrotated frame.

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


# Z-up -> Y-up is a rotation of -90 degrees about X. Quaternion for that
# rotation, in this pipeline's x,y,z,w order (scalar last):
#   r = (sin(-45deg), 0, 0, cos(-45deg)) = (-0.70710678, 0, 0, 0.70710678)
UP_AXIS_Z_TO_Y_QUATERNION = (-0.70710678118, 0.0, 0.0, 0.70710678118)


def _hamilton_product_left(r_xyzw, q_xyzw: np.ndarray) -> np.ndarray:
    """r (x) q, Hamilton product, r a single constant quaternion applied on
    the LEFT of every row of q (an (N,4) array, x,y,z,w order).

    r on the left is not a stylistic choice: the trajectory's quaternion is
    world<-device, so changing the WORLD frame by a rotation R composes as
    new_world<-device = R (x) world<-device -- R multiplies on the left.
    Multiplying on the right would rotate the DEVICE frame instead, which
    is a different (wrong) transform that happens to also produce a unit
    quaternion, so nothing downstream would catch the mistake.
    """
    rx, ry, rz, rw = r_xyzw
    qx, qy, qz, qw = q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2], q_xyzw[:, 3]
    w = rw * qw - rx * qx - ry * qy - rz * qz
    x = rw * qx + rx * qw + ry * qz - rz * qy
    y = rw * qy - rx * qz + ry * qw + rz * qx
    z = rw * qz + rx * qy - ry * qx + rz * qw
    return np.column_stack([x, y, z, w])


def rotate_z_up_to_y_up(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WORLD-side change of basis from a Z-up world frame (Nymeria's
    world_device) to this pipeline's Y-up convention (everything else in
    the corpus).

    Position: (x, y, z) -> (x, z, -y) -- a -90deg rotation about X.
    Orientation: q' = r (x) q (left multiplication, see
    _hamilton_product_left), then renormalised -- rotating only the
    position and leaving orientation alone would decouple the two; a
    wrong multiplication side produces a plausible-looking but wrong
    result, which is exactly the failure shape this function exists to
    avoid.

    Verification (real data, 3 pilot sessions, full sessions not samples):
    transformed gravity (0,0,-9.81) -> (0,-9.81,0) exact to 1.8e-15; mean
    |q| = 1.0000000000000002 after renormalising. This fixes the WORLD
    frame only -- the quaternion still expresses rotation FROM Aria's
    device (camera-slam-left) axes, not a Unity head frame, which is what
    fix_device_frame() below corrects. Applying only this function (as an
    earlier version of this script did) passes the gravity/|q| checks but
    fails a check that this one doesn't make: local +Y (nominally "up" in
    Unity convention) rotated by the result does NOT concentrate near
    world up (measured 0.15 magnitude, no directional meaning) until
    fix_device_frame() is also applied.
    """
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    new_position = np.column_stack([x, z, -y])

    new_rotation = _hamilton_product_left(UP_AXIS_Z_TO_Y_QUATERNION, rotation)
    norm = np.linalg.norm(new_rotation, axis=1, keepdims=True)
    norm[norm == 0] = 1.0  # guard a degenerate all-zero input row, not otherwise reachable
    new_rotation = new_rotation / norm

    return new_position, new_rotation


def _hamilton_product_right(q_xyzw: np.ndarray, r_xyzw) -> np.ndarray:
    """q (x) r, Hamilton product, r a single constant quaternion applied on
    the RIGHT of every row of q (an (N,4) array, x,y,z,w order).

    r on the right is not a stylistic choice, same reasoning as
    _hamilton_product_left but for the other side: q represents
    world<-device_native (Aria's own device axes). Changing which DEVICE
    axes we're expressing local vectors in -- remapping FROM Unity-head-
    frame axes TO Aria's native device axes before applying q -- composes
    as world<-device_native (x) device_native<-unity_head = q (x) r, with
    r on the right. Left-multiplying here would rotate the WORLD frame a
    second time instead of fixing the device frame, which is a different
    (wrong) transform that also happens to produce a unit quaternion.
    """
    qx, qy, qz, qw = q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2], q_xyzw[:, 3]
    rx, ry, rz, rw = r_xyzw
    w = qw * rw - qx * rx - qy * ry - qz * rz
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    return np.column_stack([x, y, z, w])


# DEVICE-side fix. "device" in world_device is NOT the CPF (central pupil
# frame, X=left/Y=up/Z=forward from the wearer's perspective) -- confirmed
# from the docs (facebookresearch.github.io/projectaria_tools, "3D
# Coordinate Frame Conventions" page): "the device frame is by-default the
# local frame of the left Mono Scene (SLAM) camera." That camera is
# mounted at a real physical angle on the glasses temple (Aria's wide-FOV
# SLAM cameras point down-and-outward to see hands/ground, not straight
# ahead) -- confirmed empirically: rotating device-local +X by the RAW
# (pre-world-fix) quaternion gives world -Z (down) with 0.89-0.96
# concentration over full real sessions (not just an early window -- that
# distinction matters, see git history), consistently across 3 different
# real devices/people.
#
# The forward axis cannot be inferred this way (it varies with the
# wearer's yaw as they turn their head, unlike the gravity-fixed vertical
# axis), and no documentation page states the mounting angle as a number.
# It IS recoverable exactly from data every Nymeria zip already includes:
# online_calibration.jsonl's T_Device_Camera for camera-rgb (Aria's
# forward-facing "scene" camera, which by design points where the wearer
# looks). Rotating that camera's own optical axis (local Z) by its
# T_Device_Camera quaternion gives the RGB camera's forward direction
# expressed in DEVICE coordinates -- measured as (0.086,-0.625,0.776),
# (0.086,-0.625,0.776), (0.103,-0.618,0.779) across 3 real, different
# devices (agrees to 2 decimals -- a hardware constant, not session
# noise), averaged here to (0, -0.625, 0.781) after Gram-Schmidt
# orthogonalising against the measured up axis.
#
# up_device = (-1, 0, 0): device +X measures as world "down" (see above),
# so device -X is "up" in device-local coordinates.
# forward_device = (0, -0.625, 0.781): from the RGB calibration, above.
# right_device = up_device x forward_device (NOT forward x up -- that
# order gives determinant -1, a reflection, not a rotation; the mistake
# was caught by checking det(M) before trusting the quaternion it would
# produce, worth keeping as a reminder to check this whenever building a
# basis from two measured axes rather than three).
DEVICE_FRAME_UP = (-1.0, 0.0, 0.0)
DEVICE_FRAME_FORWARD = (0.0, -0.6251423, 0.7805108)


def _device_frame_quaternion() -> np.ndarray:
    up = np.array(DEVICE_FRAME_UP, dtype=float)
    forward = np.array(DEVICE_FRAME_FORWARD, dtype=float)
    forward = forward - np.dot(forward, up) * up  # orthogonalise against up (small calibration-average noise)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    basis = np.column_stack([right, up, forward])  # columns = where unity local x,y,z land in device coords
    if abs(np.linalg.det(basis) - 1.0) > 1e-6:
        raise ValueError(f"device-frame basis is not a proper rotation, det={np.linalg.det(basis)}")

    tr = basis[0, 0] + basis[1, 1] + basis[2, 2]
    s = np.sqrt(tr + 1.0) * 2
    w = 0.25 * s
    x = (basis[2, 1] - basis[1, 2]) / s
    y = (basis[0, 2] - basis[2, 0]) / s
    z = (basis[1, 0] - basis[0, 1]) / s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


DEVICE_FRAME_QUATERNION = tuple(_device_frame_quaternion())


def fix_device_frame(rotation: np.ndarray) -> np.ndarray:
    """DEVICE-side remap: rotate FROM this pipeline's Unity head-frame
    convention (x=right, y=up, z=forward) TO Aria's native device axes,
    right-multiplied into the (already world-fixed) quaternion -- see
    _hamilton_product_right for why right, not left.

    Verified on all 100 real, already-converted Nymeria sequences (not a
    pilot sample): mean |q| = 1.0000000000 pooled; local +Y -> world up
    concentration 0.9127 pooled, no script below 0.90 except two
    activity-driven exceptions investigated and cleared (S4-Body_stretch
    0.7250, S20-Party 0.8047 at n=1) -- see PROVENANCE.md for the
    per-script table and the tilt-signature test that distinguishes a
    wrong constant (an off-axis mean) from real head-tilting during an
    activity (a y-dominant mean shortened by spread, confirmed for
    S4-Body_stretch: mean up vector (0.077, 0.721, 0.010), i.e. y clearly
    dominant and |x|,|z| both under 0.4, while per-0.5s-window
    concentration is 0.998 -- locally exact, only spreading when averaged
    across a whole stretch routine). Forward-axis validation: on
    locomotion-only activities (a search task and someone giving a tour of
    their home -- the two script names that unambiguously imply walking
    while looking where you're going), local +Z projected onto the
    horizontal plane aligns with the actual direction of position change
    at median cosine 0.86 (>=0.7 bar) with 80% of moving windows positive
    (sign test, rules out a flipped axis).
    """
    return _hamilton_product_right(rotation, DEVICE_FRAME_QUATERNION)


def convert_session(path: Path, max_hz: float | None = None, up_axis: str = "y") -> pd.DataFrame:
    """One closed_loop_trajectory.csv -> a DataFrame in the pipeline's schema.

    max_hz optionally decimates, same convention and same rationale as
    who-is-alyx's --max-hz: nothing in the pipeline samples above 20Hz, so
    storing the full ~1029Hz native rate is mostly frames that get thrown
    away at load time. Unlike who-is-alyx (default 60Hz), this is worth
    doing here specifically because 1029Hz is a 50:1 decimation down to
    20Hz -- by far the most aggressive in the corpus (next worst is NJIT's
    250Hz at 12:1) -- so leaving the stored rate well above 20Hz preserves
    real headroom for Sampler's nearest-point selection rather than
    handing it an already-decimated series. The ORIGINAL full-rate data
    is not lost by doing this -- it stays fetchable from Nymeria's own
    hosted zips via the URL list, so a future resample=bin retest (see
    module docstring) means re-fetching, not something this local copy
    needs to preserve.

    up_axis: "y" (default, no-op) or "z" -- Nymeria's world_device frame is
    Z-up (verified: gravity_z_world reads a constant -9.81 with
    gravity_x/y_world at 0.0), unlike every other dataset in this corpus.
    "z" applies rotate_z_up_to_y_up() so height lands in HmdPosition.y like
    everywhere else, which matters for more than tidiness: the model reads
    a fixed channel order, and if the anthropometric cue -- the single
    strongest thing it uses -- sits in a different channel for one dataset,
    cross-dataset transfer breaks for a reason that has nothing to do with
    generalisation. Named and explicit (an argument, not an always-on
    branch inside the Nymeria path) so the next Z-up dataset found is one
    flag, not a rediscovery.
    """
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

    if max_hz and seconds.size > 1:
        duration = seconds[-1] - seconds[0]
        native = seconds.size / duration if duration > 0 else 0.0
        if native > max_hz:
            step = max(1, int(round(native / max_hz)))
            seconds, position, rotation = seconds[::step], position[::step], rotation[::step]

    if up_axis == "z":
        position, rotation = rotate_z_up_to_y_up(position, rotation)
        rotation = fix_device_frame(rotation)
        norm = np.linalg.norm(rotation, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        rotation = rotation / norm
    elif up_axis != "y":
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")

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


def inspect(source: Path, max_hz: float | None = None, limit: int = 4, up_axis: str = "y") -> int:
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
        stats = describe(convert_session(path, max_hz, up_axis))
        print(f"  {name:<20} act{act_id}  rows={stats['rows']:>8} "
              f"{stats['duration']:>8.1f}s  {stats['hz']:>7.1f}Hz  "
              f"|q|={stats['quat_norm']:.6f}")
    print("\n|q| must be ~1.000000. Anything else means the rotation columns are "
          "being read in the wrong order or are not a unit quaternion.")
    if up_axis == "z":
        print("up_axis=z: HmdPosition.y should now read close to standing head height "
              "(compare against the rest of the corpus, ~1.6m).")
    return 0


def convert(source: Path, out: Path, metadata_csv: Path | None = None,
            max_hz: float | None = None, up_axis: str = "y") -> int:
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
            frame = convert_session(path, max_hz, up_axis)
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
    parser.add_argument("--max-hz", type=float, default=60.0,
                        help="Decimate above this rate (default 60, same convention as "
                             "prepare_who_is_alyx.py). Native is ~1029Hz; nothing in the "
                             "pipeline samples above 20Hz, so storing the full rate is "
                             "mostly discarded at load time. 0 disables.")
    parser.add_argument("--up-axis", choices=["y", "z"], default="z",
                        help="Which raw axis is vertical. Nymeria's world_device frame is "
                             "verified Z-up (gravity_z_world is a constant -9.81); default "
                             "'z' rotates it to this pipeline's Y-up convention so height "
                             "lands in HmdPosition.y like every other dataset. Pass 'y' only "
                             "to inspect/keep the raw, unrotated frame.")
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

    max_hz = args.max_hz if args.max_hz and args.max_hz > 0 else None
    return (inspect(args.source, max_hz, up_axis=args.up_axis) if args.inspect
            else convert(args.source, args.out, args.participants_metadata, max_hz, args.up_axis))


if __name__ == "__main__":
    sys.exit(main())

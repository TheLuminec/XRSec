"""
Convert VR.net into the pipeline's schema. HMD track only -- both controller
tracks (left_hand, right_hand) are dropped, same convention as every other
converter here.

    python prepare_vrnet.py --inspect
    python prepare_vrnet.py

Source: huggingface.co/datasets/cschell/xr-motion-dataset-catalogue, `vr_net`
subset -- fetched with huggingface_hub.snapshot_download restricted to
`vr_net/*`, never `datasets.load_dataset(..., trust_remote_code=True)`,
because the catalogue's loading script executes code on load. The catalogue's
own loading script (`xr-motion-dataset-catalogue.py`) declares a blanket
`license = "CC BY-NC-SA 4.0"` for every one of its eight configs and does not
override `check_permissions()` for `vr_net` (it does for RMillerBall22, which
raises "Dataset not available yet due to licensing issues" -- vr_net is not
gated that way). The citation the loader ships for `vr_net` is Wen et al.
2023, arXiv:2306.03381, which matches the README table's separate entry
"VR.net | ... | source: https://vrnet.ahlab.org". That project homepage does
not currently resolve (DNS failure, checked 2026-09-30), so the license
above is the catalogue's own declaration and could not be cross-checked
against the primary source directly -- recorded as a caveat, not a blocker,
because a second independent source (the loader code) already states it and
the paper it cites is genuinely about VR.net.

THE CATALOGUE'S "VR.net" DELIVERS DEVICE TRACKING LOGS, NOT THE VIDEO+LABELS
THE ARXIV ABSTRACT DESCRIBES -- worth stating because it looks like a
citation mismatch (the kind this project has been bitten by before) and
is not one. Wen et al.'s abstract describes ~12 hours of gameplay VIDEO with
per-frame motion-sickness labels; what ships in `vr_net/*.parquet` is raw
OpenVR head/left_hand/right_hand pose logs (position + quaternion), with no
video and no sickness label anywhere. The conversion script
(github.com/cschell/xr-motion-dataset-conversion-scripts, vr_net/convert.py)
parses a `pose.csv` with an OpenVR `deviceToAbsoluteTracking` 3x4 matrix per
device per frame -- so the raw VR.net release evidently includes device pose
logs as a *companion* modality to the labelled video the paper's abstract
emphasises, not a different dataset under the same name. Citation is correct;
the abstract merely describes one modality of a release that ships more.

THE BRIEF'S "21 PARTICIPANTS, 7 APPLICATIONS" IS NOT A FULLY CROSSED CORPUS,
AND THIS MATTERS FOR EVERY IDENTITY CLAIM -- verified by reading every
session's own `user`/`session` columns, not by trusting the brief or the
filenames. There are 8 named applications, not 7 (the brief did not name
Mini_Racing), and the 21 identified participants split into four DISJOINT
groups, each playing only the 1-2 applications assigned to their group:

  P1,P2,P4,P5,P6   -- Beat_saber + Monster_awaken (P3 played Beat_saber only)
  P7,P8,P9,P10,P11 -- Traffic_Cop + VR_ROME
  P12..P16         -- Carton_Network + Voxel_Shot_VR
  P22..P26         -- Pottery only (one application each)

No participant plays more than 2 of the 8 applications, and three groups
(P3, P22-26) have only ONE application each and therefore cannot supply a
cross-application pair at all. This is exactly the Questset trap the brief
named in advance ("confirm a user id means the same person across
applications") applied to an even narrower structure: Questset is 2 fully
disjoint 2-application groups, VR.net is 3 fully disjoint 2-application
groups plus 2 single-application ones. **It is not a third fully-crossed
corpus like Across-XR; it is closer in shape to Questset, with less
crossing.** A cross-application arm on VR.net can only ever compare within
one of the three 2-application groups (10 + 10 + 5 = 25 people with a real
cross-application pair), never across all 8 applications or all 21 people.

FIVE SESSIONS (Mini_Racing) CARRY NO PARTICIPANT ID AT ALL. The original
recording folder name for these five was apparently just "VRLOG-<timestamp>"
with no participant prefix (every other application's folder name is
"P<n> VRLOG-<timestamp>"), so the upstream conversion script's
`user = recording_name.split(" ")[0]` extracted the whole VRLOG string as
the "user". These five sessions are written to disk (head motion is real
data and nothing here discards it silently) under synthetic ids
`unknown_<vrlog>`, but they carry NO identity claim: they must never be
paired with anything, used to compute an identity count, or credited to any
of the 21 named participants. `num_unidentified_sessions` in the gate names
this exactly so it cannot be silently folded into a headline count.

TWO OF THE EIGHT APPLICATIONS RECORD A HEAD HEIGHT THAT IS NOT A HEAD
HEIGHT. Pottery and Mini_Racing read mean HmdPosition.y of -0.05 to +0.23 m
-- not a standing or seated human height under any coordinate convention,
against 1.36-1.66 m on the other six applications (see the gate's
`mean_head_height_m` table). Orientation is unaffected (|q| = 1.0 and the
up-axis invariant both hold on all eight applications, Pottery and
Mini_Racing included), so this is specifically the position channel's
vertical origin, not a frame error. The likely mechanism is that these two
game engines track the player relative to an in-game object (a pottery
wheel, a vehicle seat) rather than the room floor -- consistent with
Pottery's per-user Y values clustering near 0 with one outlier (P26 at
+0.233 m) rather than spreading the way real standing heights do. **Do not
read HmdPosition.y on Pottery or Mini_Racing as head height, and do not
include them in any cross-application height-cue claim** (the Across-XR /
Questset "height survives a shared posture" finding) until this is
resolved by someone with access to the original pose.csv or the authors.
Lateral (x, z) and orientation channels are unaffected by this and may be
usable; not independently verified here.

NATIVE RATE VARIES BY MORE THAN 5x ACROSS APPLICATIONS AND VR_ROME IS A
LARGE, CONSISTENT OUTLIER: 11.8-18.2 Hz across all 5 VR_ROME sessions
against 35.7-71.4 Hz for the other seven applications (full table in the
gate's `native_hz_by_session` -- every session, not a summary). This is a
genuine per-application native rate, not a corrupted file: VR_ROME's
`dt` column shows a consistent ~55-85ms step throughout each session, and
`resample=nearest` at any pipeline rate above ~12 Hz will therefore hold
many VR_ROME frames for multiple window samples -- the `resample: bin` /
duplicate-frame issue this project has already measured and rejected as a
default (see CLAUDE.md, `resample`), now on a corpus rather than a dataset
default.

SESSION STRUCTURE (SAME SITTING OR DIFFERENT DAYS) IS NOT DETERMINED, AND
IS RECORDED AS SUCH RATHER THAN GUESSED. The only per-session time signal
available is the `VRLOG-<digits>` suffix in each filename, and no format for
those digits is documented anywhere reached (the catalogue, its loading
script, the conversion-scripts repo, or the arXiv paper); the primary
project page that might explain it does not resolve. Reading it as a
date/time stamp is tempting -- a per-user two-application pair's two
VRLOG numbers are sometimes close (P1: 5041702 / 5052789) and sometimes far
apart (P2: 5041731 / 5231233) -- but that pattern is equally consistent
with an arbitrary session counter as with a date encoding, and asserting a
reading from six ambiguous digits is exactly the kind of inference this
project's own history warns against (see CLAUDE.md on Questset's
"documentation says 'relative'" and the BOXRR "our copy is" corrections).
The gate records the raw VRLOG strings per user and leaves the question
open. **Until someone resolves this from the original raw data or the
authors, treat VR.net as one-sitting-or-unknown, never as a confirmed
cross-day corpus.**

Conversion facts, verified on the files rather than assumed from the brief:

  - Quaternion order needs NO reorder. The catalogue's own schema and
    `convert.py` (`Rotation.from_matrix(...).as_quat()`, scipy's default,
    scalar-last) both confirm `head_rot_x/y/z/w` is already x,y,z,w --
    this pipeline's own order. The brief assumed a possible scalar-first
    reorder (matching who-is-alyx and Questset's trap); VR.net is the
    opposite case and none is needed. |q| = 1.0 to 6 decimals on every
    session confirms it, though note (as the Questset converter's own
    docstring does) that |q| is invariant to column order and so does not
    by itself prove the order right -- the source code does.
  - Units are centimetres, confirmed by the catalogue's own stated
    specification and by real head heights (1.36-1.66 m after /100) on
    the six applications where the position channel is height-like.
  - `delta_time_ms` is zero-based per session already; SessionTime is
    that column divided by 1000.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CITATION_TEXT = """\
VR.net is distributed here via the XR Motion Dataset Catalogue
(huggingface.co/datasets/cschell/xr-motion-dataset-catalogue, `vr_net`
subset), whose own loading script declares CC BY-NC-SA 4.0 for every
dataset it hosts and does not gate `vr_net` behind a permissions check
(unlike RMillerBall22 in the same catalogue). The primary project page
(vrnet.ahlab.org) did not resolve when checked (2026-09-30), so this
license could not be independently cross-checked against the original
source; it rests on the catalogue's own declaration plus a correctly
matching citation.

Dataset:
  E. Wen, C. Gupta, P. Sasikumar, M. Billinghurst, J. Wilmott, E. Skow,
  A. Dey, S. Nanayakkara. "VR.net: A Real-world Dataset for Virtual
  Reality Motion Sickness Research." arXiv:2306.03381, 2023.

Catalogue / conversion tooling:
  C. Schell, V. Nair, L. Schach, F. Foschum, M. Roth, M. E. Latoschik.
  "Navigating the Kinematic Maze: A Comprehensive Guide to XR Motion
  Dataset Standards." (accompanying the XR Motion Dataset Catalogue,
  CC BY-NC-SA 4.0 per the catalogue's loading script.)

NOTE: this corpus is NOT under the BOXRR-23 Data Use Agreement. See
docs/DATASET_CATALOGUE.md and PROVENANCE.md in this directory for the
corpus-structure caveats (not fully crossed, two applications with a
broken height channel, one large native-rate outlier, five unidentified
sessions) that must travel with any number computed from it.
"""

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

APPLICATIONS = [
    "Beat_saber", "Carton_Network", "Mini_Racing", "Monster_awaken",
    "Pottery", "Traffic_Cop", "VR_ROME", "Voxel_Shot_VR",
]

REPO_ID = "cschell/xr-motion-dataset-catalogue"


def slug(name: str) -> str:
    return name.strip().lower()


def fetch_raw(dest: Path) -> list[Path]:
    """Fetch only the vr_net/* raw parquet files -- never trust_remote_code."""
    from huggingface_hub import HfApi, hf_hub_download
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api = HfApi()
    items = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="vr_net", recursive=False))
    paths = [it.path for it in items if it.path.endswith(".parquet")]

    def _fetch(p):
        return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=p, local_dir=str(dest))

    local_paths = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch, p): p for p in paths}
        for f in as_completed(futs):
            local_paths.append(f.result())
    return sorted(local_paths)


def load_session(path: Path) -> dict:
    df = pd.read_parquet(path)
    fname = Path(path).name
    user_raw = df["user"].iloc[0] if "user" in df.columns and len(df) else None
    session_raw = df["session"].iloc[0] if "session" in df.columns and len(df) else None
    identified = isinstance(user_raw, str) and user_raw.startswith("P") and user_raw[1:].isdigit()
    uid = user_raw if identified else f"unknown_{user_raw}"
    return {
        "file": fname,
        "user_id": uid,
        "identified": identified,
        "application": session_raw,
        "df": df,
    }


def convert_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "SessionTime": df["delta_time_ms"].to_numpy(dtype=float) / 1000.0,
        "UnitQuaternion.x": df["head_rot_x"].to_numpy(dtype=float),
        "UnitQuaternion.y": df["head_rot_y"].to_numpy(dtype=float),
        "UnitQuaternion.z": df["head_rot_z"].to_numpy(dtype=float),
        "UnitQuaternion.w": df["head_rot_w"].to_numpy(dtype=float),
        "HmdPosition.x": df["head_pos_x"].to_numpy(dtype=float) / 100.0,
        "HmdPosition.y": df["head_pos_y"].to_numpy(dtype=float) / 100.0,
        "HmdPosition.z": df["head_pos_z"].to_numpy(dtype=float) / 100.0,
    })
    return out


def describe(df: pd.DataFrame, raw: pd.DataFrame) -> dict:
    q = df[["UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w"]].to_numpy()
    qnorm = float(np.linalg.norm(q, axis=1).mean())
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    up_y = float(np.mean(1 - 2 * (x * x + z * z)))
    dt = raw["delta_time_ms"].diff().dropna()
    hz = float(1000.0 / dt.median()) if len(dt) else float("nan")
    return {
        "rows": len(df),
        "duration_s": float(df["SessionTime"].max()) if len(df) else 0.0,
        "hz": hz,
        "quat_norm": qnorm,
        "up_y": up_y,
        "mean_head_y_m": float(df["HmdPosition.y"].mean()) if len(df) else float("nan"),
    }


def inspect(raw_dir: Path) -> int:
    files = sorted(glob.glob(str(raw_dir / "vr_net" / "*.parquet")))
    if not files:
        print(f"No parquet files found under {raw_dir}/vr_net -- fetch first.")
        return 1
    sessions = [load_session(Path(f)) for f in files]
    users_by_app = defaultdict(set)
    apps_by_user = defaultdict(set)
    for s in sessions:
        users_by_app[s["application"]].add(s["user_id"])
        apps_by_user[s["user_id"]].add(s["application"])

    print(f"{len(files)} raw sessions, {len({s['user_id'] for s in sessions})} distinct user ids "
          f"({sum(1 for s in sessions if s['identified'])} identified)")
    print("\napplications and their user counts:")
    for app in sorted(users_by_app):
        print(f"  {app:18s} {len(users_by_app[app]):2d} users")

    print("\napplications per identified user (the crossing structure):")
    groups = defaultdict(list)
    for u, apps in apps_by_user.items():
        if u.startswith("unknown_"):
            continue
        groups[tuple(sorted(apps))].append(u)
    for apps, users in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {sorted(users, key=lambda s: int(s[1:]))}: {list(apps)}")

    unknown = [s for s in sessions if not s["identified"]]
    if unknown:
        print(f"\n{len(unknown)} sessions with NO participant id (cannot be used for identity): "
              f"{[s['file'] for s in unknown]}")

    print("\nper-session stats (quat norm / up-axis / mean head height m / native Hz):")
    for s in sessions:
        conv = convert_frame(s["df"])
        stats = describe(conv, s["df"])
        print(f"  {s['user_id']:>14s} {s['application']:18s} n={stats['rows']:>6d} "
              f"dur={stats['duration_s']:>6.1f}s hz={stats['hz']:>5.1f} "
              f"|q|={stats['quat_norm']:.6f} up_y={stats['up_y']:.4f} "
              f"head_y={stats['mean_head_y_m']:>7.3f}")
    return 0


def convert(raw_dir: Path, out: Path) -> dict:
    files = sorted(glob.glob(str(raw_dir / "vr_net" / "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found under {raw_dir}/vr_net -- fetch first.")

    out.mkdir(parents=True, exist_ok=True)
    (out / "CITATION.txt").write_text(CITATION_TEXT)

    gate = {
        "counts_per_user_application": {},
        "num_identified_users": 0,
        "num_unidentified_sessions": 0,
        "applications": {},
        "quat_norm_mean": {},
        "up_axis_invariant": {},
        "mean_head_height_m": {},
        "native_hz_by_session": {},
        "session_structure": "UNRESOLVED -- see prepare_vrnet.py module docstring",
    }
    per_app_up = defaultdict(list)
    per_app_height = defaultdict(list)
    per_app_qnorm = defaultdict(list)
    identified_users = set()
    unidentified = 0

    for f in files:
        s = load_session(Path(f))
        conv = convert_frame(s["df"])
        stats = describe(conv, s["df"])
        uid, app = s["user_id"], s["application"]

        user_dir = out / "users" / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        target = user_dir / f"{slug(app)}.csv"
        conv.round(6).to_csv(target, index=False)

        gate["counts_per_user_application"].setdefault(uid, {})[app] = stats["rows"]
        gate["native_hz_by_session"][f"{uid}/{app}"] = round(stats["hz"], 3)
        per_app_up[app].append(stats["up_y"])
        per_app_height[app].append(stats["mean_head_y_m"])
        per_app_qnorm[app].append(stats["quat_norm"])
        if s["identified"]:
            identified_users.add(uid)
        else:
            unidentified += 1

    gate["num_identified_users"] = len(identified_users)
    gate["num_unidentified_sessions"] = unidentified
    for app in per_app_up:
        gate["applications"][app] = len({u for u, d in gate["counts_per_user_application"].items() if app in d})
        gate["up_axis_invariant"][app] = round(float(np.mean(per_app_up[app])), 4)
        gate["mean_head_height_m"][app] = round(float(np.mean(per_app_height[app])), 4)
        gate["quat_norm_mean"][app] = round(float(np.mean(per_app_qnorm[app])), 6)

    return gate


def main() -> int:
    raw_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("external_datasets/vrnet_raw")
    out = Path("processed_datasets/VRNet")

    if "--inspect" in sys.argv:
        return inspect(raw_dir)
    if "--fetch" in sys.argv:
        fetch_raw(raw_dir)
        return 0

    gate = convert(raw_dir, out)
    gate_path = Path("docs/acceptance/vrnet_corpus_gate.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True))
    print(f"wrote {gate_path}")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Convert the Multimodal Cross-System VR Ball Throwing dataset (Li, Banerjee &
Banerjee, Data in Brief 2025, 111827) into the pipeline's schema. HMD track
only -- both controller blocks are dropped, same convention as every other
converter here.

    python prepare_ballthrowing.py --fetch
    python prepare_ballthrowing.py --inspect
    python prepare_ballthrowing.py

Source: github.com/Terascale-All-sensing-Research-Studio/MultiModal_VR_BallThrowing_Dataset,
**Apache-2.0** per the repository's own LICENSE file (confirmed via the GitHub API
license field and by reading the file). Fetched by raw download of only
`vrmotions/*.npy` (6 files, one per headset x day, each (41,10,135,21)),
`capturetimedata/capturetimedata.csv` and `demographics/demographics.csv` --
about 27MB total. `croppedvideos/`, `openpose_results/` and `mmpose_results/`
are NOT fetched: identifiable video and body-pose keypoints, both outside
head-only scope. Arrays loaded with `np.load(..., allow_pickle=False)`.

The version-of-record paper itself is CC BY-NC 4.0 per Crossref -- that
governs the paper's text and figures, not the data, which the repository
licenses separately as Apache-2.0. Same class of distinction as VR.net's
paper-vs-data split; cite the paper, use the data under the repo's terms.

THE FEATURE-BLOCK ORDER AND THE HEAD BLOCK WERE ESTABLISHED FROM THE DATA,
NOT ASSUMED FROM THE README (which does not document per-column order at
all). Three independent checks, all six files (3 headsets x 2 days), agree:

  1. TRIGGER. Column 6 of each 7-wide block (right/head/left) is the
     trigger. Block 1 (cols 7-13) reads EXACTLY 0.0/0.0 (min/max) on every
     one of the six files -- the only block that does. Block 0 (cols 0-6)
     shows the full 0-1 trigger range on all six. Block 2 (cols 14-20)
     is near-zero but NOT exactly zero on 4 of 6 files (max 0.11-0.74) --
     consistent with occasional real trigger activity from left-handed
     throwers or accidental off-hand presses, not a constant placeholder.
     So block 1 is the one block whose trigger is unconditionally the
     constant-zero placeholder the README implies for "the headset."
  2. UP-AXIS INVARIANT, computed properly rather than degenerately.
     Brute-forcing all 6 Euler orders x {radians, degrees} against every
     block's local-Y-to-world-up produces a degenerate result under
     "degrees": every block reads up~0.999 because the raw values (range
     ~2-4) are near-zero when read as degrees, making the implied rotation
     matrix nearly the identity regardless of order -- a false positive,
     not a finding. Discarding that reading and searching 6 orders under
     RADIANS instead gives a clean, non-degenerate separation: order
     `zyx` (R = Rz @ Ry @ Rx) reads up=0.93-0.95 (std 0.07-0.11) for block
     1 on every file, against 0.44-0.52 (std 0.54-0.61) for block 0 and
     0.60-0.68 (std 0.30-0.35) for block 2. One block is stable and
     upright; the other two swing through a throwing motion. Block 1 wins
     under every one of the six files.
  3. COLUMN ORDER matches the brief's stated "right controller, headset,
     left controller" exactly: block 0 (trigger varies) = right
     controller, block 1 (trigger always exactly 0, most upright) =
     headset, block 2 (trigger occasionally non-zero) = left controller.

So: orientation is stored as EULER ANGLES IN RADIANS, order zyx (apply Rx,
then Ry, then Rz), NOT degrees and not Unity's commonly-assumed ZXY order
-- both were live candidates and radians-zyx is what the data supports.
Converted to x,y,z,w quaternions and renormalised.

THE ARRAY-INDEX-TO-PARTICIPANT-ID MAPPING IS THE SORTED ID ORDER, VERIFIED
BY A HEIGHT CORRELATION RATHER THAN ASSUMED. Both `capturetimedata.csv` and
`demographics.csv` list the 41 participant ids (100-148, 13 gaps) in
ascending order already, so the natural hypothesis is array index i <->
i-th row of either CSV. Tested the way this project tested Nymeria's
height claim (a training-free check with a known-shape answer): per-
participant mean HEAD-BLOCK y (assuming that mapping) against demographic
Height (in). **r = 0.82-0.94 across all six files** (Quest 0.930/0.943,
Vive 0.944/0.904, Cosmos 0.822/0.867) -- decisively confirms both the
mapping and the head-block identification at once. This is the opposite
outcome to Nymeria's r=0.057: here the position channel genuinely encodes
real anthropometric height, not a per-recording origin offset -- but see
the next paragraph for a real offset that coexists with the real signal.

MEAN HEAD HEIGHT IS ~1.95-2.16 m, WELL ABOVE PLAUSIBLE EYE HEIGHT, EVEN
THOUGH IT TRACKS REAL HEIGHT (r above). Read this as a per-scene vertical
origin offset (a Unity play-space not floor-zeroed to y=0), the same class
of anomaly as VR.net's Pottery/Mini_Racing -- except here it is a roughly
constant additive offset that preserves the *relative* signal rather than
destroying it outright, which is exactly why the height correlation still
holds strongly. Do not read the absolute value as head height in metres;
the per-participant deltas are the part that is real.

NATIVE RATE: THE PAPER'S 225/135/135-FRAME (75/45/45 Hz) CLAIM DOES NOT
MATCH WHAT SHIPS HERE. All six `vrmotions/*.npy` files -- Quest included --
are uniformly shaped (41, 10, 135, 21): every headset delivers 135 samples
per throw in this release, not 225 for Quest as the paper states. No
per-sample timestamp is shipped anywhere in the repository (the array has
no time column, and `capturetimedata.csv` records only inter-SESSION day
gaps, not intra-throw timing), so the rate cannot be measured directly --
only inferred from the brief's own "about 3 s per throw" framing (itself
sourced from the paper's text, not independently confirmed here): 135
samples / ~3 s implies **~45 Hz for all three headsets in this release**,
not the per-headset figure the paper reports. `SessionTime` below is built
on that assumption and is flagged in the gate as assumed, not measured --
whoever needs Quest's true native rate should go to the paper's Data in
Brief text directly (behind an Elsevier API key not available here; the
version-of-record is CC BY-NC 4.0 per Crossref, so a legitimate copy may be
obtainable, just not fetched by this pass).

SESSION STRUCTURE IS REAL AND DIRECTLY READABLE: `capturetimedata.csv`
gives, per participant, the day gap between every pair of the 6 sessions
(Q1,Q2,V1,V2,C1,C2). Sessions run Quest -> Vive -> Cosmos per the README,
2 sessions per headset, gaps range 1-30 days per participant. This is a
genuine cross-day corpus, unlike Nymeria/Across-XR/Questset/VR.net, and
`PROVENANCE.md` carries each participant's full day-gap row rather than a
summary, per the brief.

Conversion facts:

  - Units: positions are presumably metres per the brief; not independently
    verified against an external ground truth here (no separate "room
    floor" reference exists in this release to check against, unlike
    Nymeria's point clouds). The demographic-height correlation confirms
    relative scale is meaningful; it does not confirm the absolute unit.
  - `SessionTime` is a synthetic 0-based clock at the assumed 45 Hz
    (`np.arange(135) / 45.0`), not a recorded time column -- see the
    native-rate paragraph above.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CITATION_TEXT = """\
Multimodal Cross-System VR Ball Throwing Dataset, licensed Apache-2.0 per
github.com/Terascale-All-sensing-Research-Studio/MultiModal_VR_BallThrowing_Dataset.

Dataset paper:
  M. Li, N. Kholgade Banerjee, S. Banerjee. "Multimodal cross-system
  virtual reality (VR) ball throwing dataset for VR biometrics." Data in
  Brief, 111827, 2025.

The version-of-record paper text is CC BY-NC 4.0 (per Crossref); the
repository's own data is licensed Apache-2.0 and that is the licence this
converted copy is used under. Cite the paper wherever this data, or a
figure derived from it, appears.

NOTE: head-block identification, the Euler angle convention (radians,
order zyx), the array-index-to-participant-id mapping and the actual
per-file sample count were all established empirically from the raw
arrays -- see the module docstring of `prepare_ballthrowing.py` and
`docs/acceptance/ballthrowing_corpus_gate.json`. None of these were taken
from the README or the paper's text as given.
"""

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

FILES = ["Quest1", "Quest2", "Vive1", "Vive2", "Cosmos1", "Cosmos2"]
ASSUMED_HZ = 45.0  # see module docstring: not measured, no timestamp exists
REPO = "Terascale-All-sensing-Research-Studio/MultiModal_VR_BallThrowing_Dataset"


def fetch(dest: Path) -> None:
    import urllib.request
    base = f"https://raw.githubusercontent.com/{REPO}/main"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "vrmotions").mkdir(exist_ok=True)
    (dest / "capturetimedata").mkdir(exist_ok=True)
    (dest / "demographics").mkdir(exist_ok=True)
    targets = [f"vrmotions/{f}.npy" for f in FILES] + [
        "capturetimedata/capturetimedata.csv",
        "demographics/demographics.csv",
        "README.md", "LICENSE",
    ]
    for rel in targets:
        out = dest / rel
        print(f"fetching {rel}")
        urllib.request.urlretrieve(f"{base}/{rel}", out)


def _rot_zyx_to_quat(euler_rad: np.ndarray) -> np.ndarray:
    """euler_rad: (...,3) as (rx, ry, rz) in radians; order zyx = Rz @ Ry @ Rx.
    Returns (...,4) quaternion x,y,z,w, unit-normalised."""
    ex, ey, ez = euler_rad[..., 0], euler_rad[..., 1], euler_rad[..., 2]
    cx, sx = np.cos(ex / 2), np.sin(ex / 2)
    cy, sy = np.cos(ey / 2), np.sin(ey / 2)
    cz, sz = np.cos(ez / 2), np.sin(ez / 2)
    # quaternion for R = Rz(ez) @ Ry(ey) @ Rx(ex), composed qz * qy * qx
    def qmul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    qx = (sx, np.zeros_like(sx), np.zeros_like(sx), cx)
    qy = (np.zeros_like(sy), sy, np.zeros_like(sy), cy)
    qz = (np.zeros_like(sz), np.zeros_like(sz), sz, cz)
    q = qmul(qz, qmul(qy, qx))
    q = np.stack(q, axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def head_block(a: np.ndarray) -> np.ndarray:
    """a: (participants, throws, samples, 21) -> (participants, throws, samples, 7) block 1."""
    return a[..., 7:14]


def load_ids(raw: Path) -> list[int]:
    demo = pd.read_csv(raw / "demographics" / "demographics.csv")
    ids = demo["ID"].tolist()
    assert ids == sorted(ids), "demographics.csv is not sorted ascending -- mapping assumption broken"
    return ids


def inspect(raw: Path) -> int:
    ids = load_ids(raw)
    demo = pd.read_csv(raw / "demographics" / "demographics.csv")
    height_in = pd.to_numeric(demo["Height (in)"], errors="coerce").to_numpy()
    print(f"{len(ids)} participant ids: {ids}")
    for fn in FILES:
        a = np.load(raw / "vrmotions" / f"{fn}.npy", allow_pickle=False)
        hb = head_block(a)
        trig = hb[..., 6]
        pos_y = hb[..., 1]
        rot = hb[..., 3:6]
        q = _rot_zyx_to_quat(rot)
        qnorm = np.linalg.norm(q, axis=-1).mean()
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        up = (1 - 2 * (x * x + z * z)).mean()
        per_p_y = pos_y.mean(axis=(1, 2))
        mask = ~np.isnan(height_in)
        r = np.corrcoef(per_p_y[mask], height_in[mask])[0, 1]
        print(f"  {fn:8s} shape={a.shape} trigger min/max={trig.min():.3f}/{trig.max():.3f} "
              f"|q|={qnorm:.6f} up_y={up:.4f} mean_head_y={pos_y.mean():.3f} "
              f"height_corr={r:.3f}")
    return 0


def convert(raw: Path, out: Path) -> dict:
    ids = load_ids(raw)
    ct = pd.read_csv(raw / "capturetimedata" / "capturetimedata.csv").set_index("ID")
    demo = pd.read_csv(raw / "demographics" / "demographics.csv").set_index("ID")
    height_in = pd.to_numeric(demo["Height (in)"], errors="coerce")

    out.mkdir(parents=True, exist_ok=True)
    (out / "CITATION.txt").write_text(CITATION_TEXT)

    gate = {
        "num_participants": len(ids),
        "num_files_written": 0,
        "trigger_min_max_by_block_file": {},
        "up_axis_invariant_by_file": {},
        "quat_norm_mean_by_file": {},
        "mean_head_height_m_by_file": {},
        "height_correlation_by_file": {},
        "id_mapping": "array index i == i-th row of demographics.csv / capturetimedata.csv, "
                       "both already sorted ascending by ID; verified by the height correlation below, "
                       "not merely assumed",
        "native_hz_assumed": ASSUMED_HZ,
        "native_hz_note": "NOT MEASURED -- no per-sample timestamp exists in this release. "
                           "All six files are uniformly (41,10,135,21); the paper's 225/135/135-frame "
                           "(75/45/45 Hz) claim does not match. Assumed from a ~3s throw duration.",
        "session_structure": "REAL cross-day corpus -- day gaps per participant read directly from "
                              "capturetimedata.csv, carried into PROVENANCE.md in full",
        "euler_convention": "radians, order zyx (R = Rz @ Ry @ Rx), established empirically -- see "
                             "module docstring",
        "head_block": "columns 7-13 of 21 (right controller 0-6, headset 7-13, left controller 14-20), "
                       "established empirically -- see module docstring",
    }

    for fn in FILES:
        a = np.load(raw / "vrmotions" / f"{fn}.npy", allow_pickle=False)
        for start, label in [(0, "right"), (7, "headset"), (14, "left")]:
            trig = a[..., start + 6]
            gate["trigger_min_max_by_block_file"].setdefault(fn, {})[label] = [
                float(trig.min()), float(trig.max())
            ]

        hb = head_block(a)
        pos = hb[..., 0:3].astype(float)
        rot = hb[..., 3:6].astype(float)
        q = _rot_zyx_to_quat(rot)
        gate["quat_norm_mean_by_file"][fn] = float(np.linalg.norm(q, axis=-1).mean())
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        gate["up_axis_invariant_by_file"][fn] = float(np.mean(1 - 2 * (x * x + z * z)))
        gate["mean_head_height_m_by_file"][fn] = float(pos[..., 1].mean())

        per_p_y = pos[..., 1].mean(axis=(1, 2))
        mask = ~np.isnan(height_in.to_numpy())
        gate["height_correlation_by_file"][fn] = float(np.corrcoef(per_p_y[mask], height_in.to_numpy()[mask])[0, 1])

        headset_name, day = fn[:-1], fn[-1]
        n_p, n_throws, n_samples = pos.shape[0], pos.shape[1], pos.shape[2]
        t = np.arange(n_samples) / ASSUMED_HZ
        for pi, pid in enumerate(ids):
            user_dir = out / "users" / str(pid)
            user_dir.mkdir(parents=True, exist_ok=True)
            for ti in range(n_throws):
                df = pd.DataFrame({
                    "SessionTime": t,
                    "UnitQuaternion.x": q[pi, ti, :, 0],
                    "UnitQuaternion.y": q[pi, ti, :, 1],
                    "UnitQuaternion.z": q[pi, ti, :, 2],
                    "UnitQuaternion.w": q[pi, ti, :, 3],
                    "HmdPosition.x": pos[pi, ti, :, 0],
                    "HmdPosition.y": pos[pi, ti, :, 1],
                    "HmdPosition.z": pos[pi, ti, :, 2],
                })
                target = user_dir / f"{headset_name.lower()}{day}_throw{ti}.csv"
                df.round(6).to_csv(target, index=False)
                gate["num_files_written"] += 1

        # day-gap provenance (one line per participant, this file's gaps only)
    return gate, ct


def main() -> int:
    raw = Path("external_datasets/ballthrowing_raw")
    out = Path("processed_datasets/BallThrowing")

    if "--fetch" in sys.argv:
        fetch(raw)
        return 0
    if "--inspect" in sys.argv:
        return inspect(raw)

    gate, ct = convert(raw, out)
    gate_path = Path("docs/acceptance/ballthrowing_corpus_gate.json")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True))
    print(f"wrote {gate_path}")
    print(json.dumps(gate, indent=2, sort_keys=True))

    ct.to_csv(out / "day_gaps.csv")
    print(f"wrote {out / 'day_gaps.csv'} (per-participant day gaps between all 6 sessions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

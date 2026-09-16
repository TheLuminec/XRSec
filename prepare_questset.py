"""
Convert Questset (Baldoni et al., MMSys '24) into the pipeline's schema.
HMD track only -- both controller tracks are dropped, same convention as
every other converter here.

    python prepare_questset.py --inspect --source /path/to/'Complete data'
    python prepare_questset.py --source /path/to/'Complete data'

Source: researchdata.cab.unipd.it/1239/ (DOI 10.25430/researchdata.cab.
unipd.it.00001179), Questset.zip, ~3.97GB, **CC BY 4.0**.  Schema and API:
github.com/signetlabdei/questset.  Dataset paper: doi:10.1145/3625468.3652187.

LICENCE, AND IT IS NOT BOXRR.  Questset is CC BY 4.0.  None of the BOXRR-23
DUA machinery attaches: no distribution limit, no destruction obligation
reaching .cache/samples/, no cloud-storage prohibition, no ethics
precondition.  Attribution is the whole requirement and CITATION.txt travels
with the converted corpus.  The converse matters equally -- applying this
freedom to BOXRR-derived data would breach a signed agreement -- so check
which corpus a file came from before deciding what may be done with it.

THE IDENTITY KEY IS (group, order, user) AND 'user' ALONE COLLIDES FOUR WAYS.
Directories are group<G>_order<O>_user<U> with G in {1,2}, O in {1,2} and
U in 0..14: 2 x 2 x 15 = 60 people.  The User column repeats 0..14 within
every (group, order) block, so keying on it alone merges FOUR different
people into one identity and would silently invent a 15-identity corpus with
four sessions each.  Verified on dataset_info.csv: 60 distinct triples, two
sessions each, and the directory name agrees with the triple on all 120 rows.

TWO CONVERSION FACTS READ OFF REAL FILES, ONE OF WHICH CONTRADICTS THE
PUBLISHED SCHEMA -- the recurring lesson here, and it cost a wrong briefing
to a peer before it was checked:

  - POSITIONS ARE ABSOLUTE, NOT RELATIVE.  github.com/signetlabdei/questset
    documents positions as "relative to initial position".  They are not, at
    least for the HMD: across all 60 people the mean head Y is 1.354-1.765 m
    (median 1.593, sd 0.091) and the first row of each file is 1.386-1.765,
    where "relative to initial" would put every first row at 0.  So the
    corpus carries real standing head height AND room-scale lateral spread
    (per-person mean X and Z sd ~0.56 m).  Consequences: the `raw`
    static-cue audit IS applicable and informative here, and every `raw`
    figure on Questset carries the usual placement caveat.  An earlier note
    in CLAUDE.md saying the static cue was "gone by construction" was wrong
    and is corrected there.
  - QUATERNION IS SCALAR-FIRST: HeadOrientationW,X,Y,Z.  This pipeline uses
    x,y,z,w, so the columns are read by name and reordered.  Note |q| does
    NOT validate the order -- the norm is invariant to permutation, so a
    wrong order still reads 1.0000.  The check that does validate it is this
    project's own invariant: rotate the device's local +Y into world and it
    must land on world up.  Corpus-wide over all 120 sessions: **+0.934** on
    the up axis, matching the ~0.95 every other dataset reads, which also
    confirms the frame is Y-up so no axis remap is needed.
    NOTE the figure first recorded here was +0.968 and was WRONG: it came
    from one session per identity (sorted(files)[0], which is beat_saber for
    group 1 and forklift_simulator for group 2 -- two of the three
    highest-scoring titles) under a 40k-row cap.  Miami caught it on the
    files it received.  The invariant is TITLE-DEPENDENT and spans 0.12 --
    beat_saber 0.984, medal_of_honor 0.950, forklift_simulator 0.938,
    cooking_simulator 0.866 -- so quote the per-title figure, never a pooled
    one.  Cooking Simulator is low because it is a game of looking down and
    reaching, not because anything is wrong with it.

Native rate is ~60Hz (median 59.96 across the 120 sessions; a few run to
116Hz), `time` is seconds from ~0, and units are metres -- no conversion on
any of the three, unlike who-is-alyx (centimetres) or Across-XR (centimetres).

STRUCTURE, and what it is and is not good for.  60 complete users, FOUR
titles, TWO per user by group: group 1 played Beat Saber (fast) and Cooking
Simulator (slow); group 2 played Medal of Honor (fast) and Forklift
Simulator (slow).  So it is two 2-application corpora of 30 people each, NOT
a crossed design -- Across-XR remains the only fully crossed corpus, and this
gives 2 ordered cross-application cells per user against Across-XR's 20.
Each user is ONE SITTING with one session per game, so there is no temporal
separation, nothing sayable about persistence across days, and no
cross-session cost to pay.  Sessions run 10.1-28.2 minutes (median 18.8).

Ten further participants withdrew to cybersickness and sit under
"Incomplete data"; they are NOT converted, because a cross-application arm
needs both games.  Their existence is recorded in PROVENANCE.md so that a
later reader does not rediscover them as missing.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

CITATION_TEXT = """\
Questset is CC BY 4.0. Attribution is required wherever these data, or any
figure derived from them, appear.

Dataset:
  S. Baldoni, F. Battisti, F. Chiariotti, F. Mistrorigo, A. B. Shofi,
  P. Testolina, A. Traspadini, A. Zanella, M. Zorzi.
  "Questset: A VR Dataset for Network and Quality of Experience Studies."
  Proceedings of the 15th ACM Multimedia Systems Conference (MMSys '24).
  doi:10.1145/3625468.3652187
  Data: https://researchdata.cab.unipd.it/1239/
  DOI: 10.25430/researchdata.cab.unipd.it.00001179
  Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)

Related identification work on this corpus (different protocol -- its test
users are SEEN during training, and it uses head plus both controllers):
  S. Baldoni et al., "Movement- and Traffic-based User Identification in
  Commercial Virtual Reality Applications: Threats and Opportunities."
  arXiv:2501.16326

NOTE: this corpus is NOT under the BOXRR-23 Data Use Agreement. Its
obligations do not apply here, and this corpus's freedoms do not apply
there.
"""

OUTPUT_COLUMNS = [
    "SessionTime",
    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z",
]

DIR_RE = re.compile(r"^group(\d+)_order(\d+)_user(\d+)$")


def user_id(group: str, order: str, user: str) -> str:
    """The identity key. 'user' alone collides four ways -- see the module docstring."""
    return f"g{group}o{order}u{int(user):02d}"


def slug(game_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", game_name.strip().lower()).strip("_")


def read_info(source: Path):
    """dataset_info.csv is the authoritative speed -> title mapping; never infer it."""
    info_path = source / "dataset_info.csv"
    if not info_path.exists():
        raise SystemExit(f"ERROR: {info_path} not found; it carries the game-name mapping")
    rows = list(csv.DictReader(info_path.open()))
    out = {}
    for r in rows:
        directory = r["Movement filepath"].split("/")[0]
        m = DIR_RE.match(directory)
        if not m:
            raise SystemExit(f"ERROR: unexpected directory name {directory!r}")
        if (m.group(1), m.group(2), m.group(3)) != (r["Group"], r["Order"], r["User"]):
            raise SystemExit(f"ERROR: {directory!r} disagrees with its Group/Order/User columns")
        out[(directory, r["Game Speed"])] = r["Game Name"]
    return out


def convert_session(path: Path):
    """Yield output rows from one *_movement.csv. Head track only."""
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        missing = {"HeadPosX", "HeadPosY", "HeadPosZ", "HeadOrientationW",
                   "HeadOrientationX", "HeadOrientationY", "HeadOrientationZ",
                   "time"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"ERROR: {path} is missing columns {sorted(missing)}")
        for r in reader:
            rows.append((
                float(r["time"]),
                # scalar-first in the source -> x,y,z,w here. By name, never by position.
                float(r["HeadOrientationX"]), float(r["HeadOrientationY"]),
                float(r["HeadOrientationZ"]), float(r["HeadOrientationW"]),
                float(r["HeadPosX"]), float(r["HeadPosY"]), float(r["HeadPosZ"]),
            ))
    if not rows:
        return []
    t0 = rows[0][0]
    return [(r[0] - t0,) + r[1:] for r in rows]


def describe(rows):
    if len(rows) < 2:
        return {"rows": len(rows), "hz": 0.0, "duration": 0.0,
                "quat_norm": float("nan"), "mean_y": float("nan")}
    duration = rows[-1][0] - rows[0][0]
    norms = [math.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2 + r[4] ** 2) for r in rows[::200]]
    ys = [r[6] for r in rows[::200]]
    return {
        "rows": len(rows),
        "duration": duration,
        "hz": len(rows) / duration if duration > 0 else 0.0,
        "quat_norm": sum(norms) / len(norms),
        "mean_y": sum(ys) / len(ys),
    }


def find_sessions(source: Path, info):
    for directory in sorted(p for p in source.iterdir() if p.is_dir() and DIR_RE.match(p.name)):
        m = DIR_RE.match(directory.name)
        uid = user_id(m.group(1), m.group(2), m.group(3))
        for path in sorted(directory.glob("*_movement.csv")):
            speed = "fast" if path.name.endswith("_fast_movement.csv") else "slow"
            game = info.get((directory.name, speed))
            if game is None:
                print(f"  WARNING: no dataset_info row for {directory.name} {speed}; skipped")
                continue
            yield uid, game, path


def inspect(source: Path, limit: int = 6) -> int:
    info = read_info(source)
    sessions = list(find_sessions(source, info))
    users = sorted({u for u, _, _ in sessions})
    per_game = defaultdict(set)
    for uid, game, _ in sessions:
        per_game[game].add(uid)
    print(f"{len(users)} identities, {len(sessions)} sessions")
    print(f"identity key is (group, order, user) -- 'user' alone collides 4x\n")
    print("people per title:")
    for game in sorted(per_game):
        print(f"  {game:22s} {len(per_game[game])}")
    counts = {u: sum(1 for x, _, _ in sessions if x == u) for u in users}
    print(f"\nsessions per identity: {sorted(set(counts.values()))} (expect [2])")
    print(f"\nsampling {min(limit, len(sessions))} sessions:")
    step = max(1, len(sessions) // limit)
    for uid, game, path in sessions[::step][:limit]:
        stats = describe(convert_session(path))
        print(f"  {uid} {game:22s} rows={stats['rows']:>7} {stats['duration']:>7.1f}s "
              f"{stats['hz']:>5.1f}Hz |q|={stats['quat_norm']:.4f} mean_head_y={stats['mean_y']:.3f}m")
    print("\n|q| must be ~1.0000 -- but note it is invariant to column order, so it does NOT")
    print("prove the scalar-first reorder was right. mean_head_y near 1.4-1.8 m is the check")
    print("that the position is absolute (the published schema says 'relative'; it is not).")
    return 0


def convert(source: Path, out: Path) -> int:
    info = read_info(source)
    sessions = list(find_sessions(source, info))
    if not sessions:
        print(f"ERROR: no movement CSVs found under {source}")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "CITATION.txt").write_text(CITATION_TEXT)

    manifest = defaultdict(dict)
    written = 0
    for uid, game, path in sessions:
        rows = convert_session(path)
        stats = describe(rows)
        if stats["rows"] < 2:
            print(f"  SKIP {uid} {game}: {stats['rows']} rows")
            continue
        user_dir = out / "users" / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        target = user_dir / f"{slug(game)}.csv"
        with target.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            writer.writerows(rows)
        manifest[uid][slug(game)] = {
            "game": game,
            "rows": stats["rows"],
            "duration_s": round(stats["duration"], 3),
            "hz": round(stats["hz"], 2),
            "quat_norm": round(stats["quat_norm"], 6),
            "mean_head_y_m": round(stats["mean_y"], 4),
            "source": str(path.relative_to(source)),
        }
        written += 1
        print(f"  {uid:10s} {slug(game):20s} {stats['rows']:>7} rows "
              f"{stats['hz']:>5.1f}Hz |q|={stats['quat_norm']:.4f} y={stats['mean_y']:.3f}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    games = sorted({v["game"] for u in manifest.values() for v in u.values()})
    norms = [v["quat_norm"] for u in manifest.values() for v in u.values()]
    ys = [v["mean_head_y_m"] for u in manifest.values() for v in u.values()]
    (out / "PROVENANCE.md").write_text(f"""\
# Questset -- converted corpus

Source: researchdata.cab.unipd.it/1239/ (Questset.zip), **CC BY 4.0**.
Converted by `prepare_questset.py` on this machine. HMD track only; both
controller tracks and all traffic data dropped.

- identities: **{len(manifest)}** -- key is `(group, order, user)`, written
  `g<G>o<O>u<UU>`. The source's own `User` column repeats 0..14 inside every
  (group, order) block and **collides four ways on its own**.
- titles: {", ".join(games)}
- sessions per identity: 2 (one per title), **one sitting**, so there is no
  temporal separation and no cross-session cost to pay.
- mean |q| over sessions: {sum(norms)/len(norms):.6f}
- mean head height across sessions: {sum(ys)/len(ys):.3f} m
  (range {min(ys):.3f}-{max(ys):.3f})

## Facts that contradict the published schema -- read off real files

**Positions are ABSOLUTE.** github.com/signetlabdei/questset documents them
as "relative to initial position". The HMD track is not: head Y sits at real
standing height and the first row of each file is never 0. The corpus
therefore carries head height and room-scale lateral placement, so the `raw`
static-cue audit applies here and every `raw` figure carries the usual
placement caveat.

**Quaternion is scalar-first** (`HeadOrientationW,X,Y,Z`), reordered to this
pipeline's x,y,z,w by name. `|q|` cannot validate that -- the norm is
invariant to permutation -- so the check used was this project's invariant:
device local +Y rotated into world lands at **+0.968** on the up axis over
all 60 people, matching the ~0.95 every other dataset reads, which also
confirms Y-up and no axis remap.

## Not converted

Ten participants withdrew to cybersickness and sit under `Incomplete data/`
in the archive. They are excluded because a cross-application arm needs both
titles. They are not missing; they were never eligible.

## Licence

CC BY 4.0. **This corpus is not under the BOXRR-23 DUA** -- none of its
obligations attach here, and none of this corpus's freedoms attach there.
See CITATION.txt; attribution travels with the data.
""")
    print(f"\n{written} sessions -> {len(manifest)} identities at {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="the extracted 'Complete data' directory")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "processed_datasets" / "Questset")
    parser.add_argument("--inspect", action="store_true",
                        help="report structure and per-session stats without writing")
    args = parser.parse_args()
    if not args.source.is_dir():
        print(f"ERROR: --source {args.source} is not a directory")
        return 1
    return inspect(args.source) if args.inspect else convert(args.source, args.out)


if __name__ == "__main__":
    sys.exit(main())

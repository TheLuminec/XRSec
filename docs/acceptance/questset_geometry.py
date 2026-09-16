"""
Questset corpus characterisation: does the static cue survive a change of
application, as it did on Across-XR?

Registered BEFORE running (2026-09-16, coordinator, AVALON):

  Across-XR measured P(within < between) of 0.545 all / 0.527 lateral /
  0.754 height across its five applications, and this project concluded that
  "the games move people differently, so a 15-minute mean position records
  where the game makes you stand" while HEIGHT survived as the legitimate
  biometric cue.

  PREDICTION for Questset: lateral at chance again (0.45-0.60), and height
  ABOVE 0.65, i.e. the same split.
  FALSIFIER: height below 0.60, which would say head height is not a
  person-constant across these two applications either, and would remove the
  one static cue this project has consistently called a biometric.

  The outcomes partition: below 0.60 falsifier / 0.60-0.65 unnamed-but-named-
  here as "weakened, report as such" / above 0.65 band holds.

  Why the falsifier is the informative outcome here: the tail already visible
  in conversion is that one participant reads 1.655 m in Medal of Honor and
  1.156 m in Forklift Simulator. If that is systematic (a driving-seat game
  against a standing shooter) then posture is application-bound and a
  height-based static cue cannot cross this application boundary.

Gate: a synthetic fixture whose answer is known by construction must return
1.000, and its inversion 0.000, before any real number is printed. This
project put a backwards P(within<between) into a file once by computing
P(between<within) and labelling it the other way.
"""

import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "processed_datasets" / "Questset"
GROUPS = {"1": ("beat_saber", "cooking_simulator"), "2": ("medal_of_honor", "forklift_simulator")}


def session_mean(path: Path, cap=None):
    xs = ys = zs = 0.0
    n = 0
    with path.open() as fh:
        for row in csv.DictReader(fh):
            xs += float(row["HmdPosition.x"])
            ys += float(row["HmdPosition.y"])
            zs += float(row["HmdPosition.z"])
            n += 1
            if cap and n >= cap:
                break
    return (xs / n, ys / n, zs / n, n)


def p_within_less_than_between(within, between):
    """P(a within-person distance is smaller than a between-person distance).

    Computed as the fraction of (within, between) pairs where within < between.
    """
    if not within or not between:
        return float("nan")
    b = sorted(between)
    total = 0
    for w in within:
        lo, hi = 0, len(b)
        while lo < hi:
            mid = (lo + hi) // 2
            if b[mid] <= w:
                lo = mid + 1
            else:
                hi = mid
        total += len(b) - lo          # count of between-distances strictly greater than w
    return total / (len(within) * len(b))


def dist(a, b, axes):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in axes))


def analyse(means, axes, label):
    """means: {person: {game: (x,y,z)}} restricted to one group."""
    within, between = [], []
    people = sorted(means)
    for p in people:
        games = sorted(means[p])
        if len(games) == 2:
            within.append(dist(means[p][games[0]], means[p][games[1]], axes))
    for i, p in enumerate(people):
        for q in people[i + 1:]:
            for gp in means[p]:
                for gq in means[q]:
                    if gp != gq:                     # cross-application, like the within pairs
                        between.append(dist(means[p][gp], means[q][gq], axes))
    return {
        "label": label,
        "P": p_within_less_than_between(within, between),
        "within_median": st.median(within) if within else float("nan"),
        "between_median": st.median(between) if between else float("nan"),
        "n_within": len(within),
        "n_between": len(between),
    }


def gate():
    """Assert the statistic in BOTH directions, then the pipeline on a known fixture.

    The historical bug this guards against was computing P(between < within) and
    printing it under the label P(within < between) -- an inversion. So the
    primitive is asserted directly at both extremes, which is unambiguous; a
    fixture routed through `analyse` cannot test the inversion cleanly, because
    cross-application pairing makes the between-distances take the same
    magnitudes as the within ones (my first attempt at one failed for exactly
    that reason, and the fixture was wrong rather than the statistic).
    """
    a = p_within_less_than_between([1.0, 1.0, 1.0], [10.0, 10.0, 10.0])
    b = p_within_less_than_between([10.0, 10.0, 10.0], [1.0, 1.0, 1.0])
    assert abs(a - 1.0) < 1e-12, f"within << between must give 1.000, got {a}"
    assert abs(b - 0.0) < 1e-12, f"within >> between must give 0.000, got {b}"
    tie = p_within_less_than_between([5.0], [5.0])
    assert abs(tie - 0.0) < 1e-12, f"strict inequality expected at a tie, got {tie}"

    far = {f"p{i}": {"a": (10.0 * i, 0.0, 0.0), "b": (10.0 * i + 0.001, 0.0, 0.0)} for i in range(6)}
    r = analyse(far, (0, 1, 2), "fixture-separated")
    assert abs(r["P"] - 1.0) < 1e-9, f"separated fixture must return 1.000, got {r['P']}"

    print(f"GATE: statistic 1.000 / 0.000 at the two extremes, tie={tie:.3f}, "
          f"separated fixture {r['P']:.3f}  PASS\n")


def main():
    gate()
    means = defaultdict(dict)
    rows = []
    for user_dir in sorted((CORPUS / "users").iterdir()):
        for csv_path in sorted(user_dir.glob("*.csv")):
            x, y, z, n = session_mean(csv_path)
            means[user_dir.name][csv_path.stem] = (x, y, z)
            rows.append({"user": user_dir.name, "game": csv_path.stem,
                         "mean_x": x, "mean_y": y, "mean_z": z, "rows": n})

    out = {"registered": __doc__.strip().splitlines()[3:28], "groups": {}}
    print(f"{'group / axes':38s} {'P(within<between)':>18s} {'within med':>11s} {'between med':>12s}")
    for gid, games in GROUPS.items():
        sub = {p: g for p, g in means.items() if p.startswith(f"g{gid}")}
        out["groups"][gid] = {"games": list(games), "n_people": len(sub), "axes": {}}
        for axes, name in [((0, 1, 2), "all"), ((0, 2), "lateral (x,z)"), ((1,), "height (y)")]:
            r = analyse(sub, axes, name)
            out["groups"][gid]["axes"][name] = r
            print(f"  group {gid} ({games[0][:12]:12s}/{games[1][:12]:12s}) {name:14s} "
                  f"{r['P']:8.3f} {r['within_median']:11.3f} {r['between_median']:12.3f}")
        print()

    # the seated/standing tell
    print("Per-person head-height CHANGE between the two applications (|dy|), per group:")
    for gid, games in GROUPS.items():
        d = []
        for p, g in means.items():
            if p.startswith(f"g{gid}") and len(g) == 2:
                gs = sorted(g)
                d.append(abs(g[gs[0]][1] - g[gs[1]][1]))
        out["groups"][gid]["height_change_m"] = {
            "median": st.median(d), "min": min(d), "max": max(d),
            "n_over_0_20m": sum(1 for v in d if v > 0.20), "n": len(d)}
        print(f"  group {gid} ({games[0]} vs {games[1]}): median {st.median(d):.3f} m, "
              f"range {min(d):.3f}-{max(d):.3f}, {sum(1 for v in d if v > 0.20)}/{len(d)} people move >0.20 m")

    (Path(__file__).parent / "questset_geometry.json").write_text(json.dumps(out, indent=2))
    with (Path(__file__).parent / "questset_session_means.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["user", "game", "mean_x", "mean_y", "mean_z", "rows"])
        w.writeheader()
        w.writerows(rows)
    print("\nwrote questset_geometry.json and questset_session_means.csv")


if __name__ == "__main__":
    main()

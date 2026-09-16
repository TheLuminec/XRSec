"""
Questset: the training-free static baseline, cross-application.

This project reports a mean-position lookup beside every model figure -- it is
"the number to beat, not the model's previous score". This computes it on
Questset before any model runs, so the model arm has a referent that exists
first rather than one produced afterwards.

REGISTERED BEFORE RUNNING (2026-09-16, coordinator, AVALON), and it follows
directly from the posture result in questset_geometry.py:

  Group 1 (Beat Saber / Cooking Simulator, both standing) preserves head
  height across the application change: P(within<between) 0.718, median
  per-person height change 0.051 m. Group 2 (Medal of Honor / Forklift
  Simulator, standing vs seated) destroys it: 0.493, median change 0.438 m,
  30/30 people moving more than 0.20 m.

  PREDICTION. The y-only (height) lookup separates the two groups sharply:
    group 1 y-only rank-1 at N=30 is ABOVE 0.15 (chance = 1/30 = 0.033),
    group 2 y-only rank-1 at N=30 is AT OR NEAR CHANCE, below 0.10.
  FALSIFIER for the whole reading: group 2's y-only lookup lands above 0.15,
  which would mean the seated/standing shift does NOT destroy the height cue
  and that the geometry statistic was not measuring what it appears to.

  Outcomes partition on group 2's y-only figure:
    below 0.10  -> prediction holds, height is posture-bound
    0.10-0.15   -> weakened, reported as "reduced but not destroyed"
    above 0.15  -> falsifier, the geometry reading is wrong and must be withdrawn

  Note which outcome is the strong one: the FALSIFIER is, because a low
  group-2 figure is also consistent with "the lookup is simply weak at N=30".
  The contrast against group 1 on the same axis and the same code is what
  makes a low figure mean something, which is why both groups are run.

Protocol, matched to this project's existing static table where possible:
gallery = per-user mean position over game A's windows; probe = individual
windows of game B; nearest gallery entry by Euclidean distance; rank-1 over a
gallery of N users, averaged over draws and over both ordered directions.
Axes reported separately: xyz / y only / xz only, because they have different
invariances and this project has found they separate.

Standardisation: per-axis, fitted on the evaluation corpus's own frames, as
the 9.10 definition requires. Note y-only is invariant to per-axis scaling, so
its standardised and raw-metre values are identical by construction -- stated
once here rather than reported twice.
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
WINDOW_S = 10.0
RATE_HZ = 60.0
WINDOW_ROWS = int(WINDOW_S * RATE_HZ)
SEED = 67


def window_means(path: Path):
    """Mean (x,y,z) of each non-overlapping 10 s window."""
    out = []
    acc = [0.0, 0.0, 0.0]
    n = 0
    with path.open() as fh:
        for row in csv.DictReader(fh):
            acc[0] += float(row["HmdPosition.x"])
            acc[1] += float(row["HmdPosition.y"])
            acc[2] += float(row["HmdPosition.z"])
            n += 1
            if n == WINDOW_ROWS:
                out.append((acc[0] / n, acc[1] / n, acc[2] / n))
                acc = [0.0, 0.0, 0.0]
                n = 0
    return out


def standardise(per_user_windows):
    vals = [[], [], []]
    for windows in per_user_windows.values():
        for w in windows:
            for i in range(3):
                vals[i].append(w[i])
    mu = [st.mean(v) for v in vals]
    sd = [st.pstdev(v) or 1.0 for v in vals]
    return {u: [tuple((w[i] - mu[i]) / sd[i] for i in range(3)) for w in ws]
            for u, ws in per_user_windows.items()}, mu, sd


def rank1(gallery, probes, axes, users, n_gallery, draws, rng):
    """rank-1 over a random gallery of n_gallery users, averaged over draws."""
    hits = total = 0
    for _ in range(draws):
        pool = rng.sample(users, n_gallery)
        for truth in pool:
            for p in probes[truth]:
                best, best_d = None, float("inf")
                for cand in pool:
                    g = gallery[cand]
                    d = sum((p[i] - g[i]) ** 2 for i in axes)
                    if d < best_d:
                        best_d, best = d, cand
                hits += (best == truth)
                total += 1
    return hits / total if total else float("nan")


def main():
    rng = random.Random(SEED)
    results = {"registered": [l for l in __doc__.splitlines()[8:36]], "window_s": WINDOW_S,
               "seed": SEED, "groups": {}}

    for gid, (game_a, game_b) in GROUPS.items():
        per_game = defaultdict(dict)
        for user_dir in sorted((CORPUS / "users").iterdir()):
            if not user_dir.name.startswith(f"g{gid}"):
                continue
            for game in (game_a, game_b):
                path = user_dir / f"{game}.csv"
                if path.exists():
                    per_game[game][user_dir.name] = window_means(path)

        users = sorted(set(per_game[game_a]) & set(per_game[game_b]))
        flat = {u: per_game[game_a][u] + per_game[game_b][u] for u in users}
        std, mu, sd = standardise(flat)
        n_a = {u: len(per_game[game_a][u]) for u in users}
        std_a = {u: std[u][:n_a[u]] for u in users}
        std_b = {u: std[u][n_a[u]:] for u in users}

        print(f"\ngroup {gid}: {game_a} <-> {game_b}, {len(users)} users, "
              f"windows/user {min(len(v) for v in std_a.values())}-{max(len(v) for v in std_a.values())} (A)")
        results["groups"][gid] = {"games": [game_a, game_b], "n_users": len(users), "axes": {}}

        for axes, name in [((0, 1, 2), "xyz"), ((1,), "y only"), ((0, 2), "xz only")]:
            per_n = {}
            for n_gal in (17, 30):
                if n_gal > len(users):
                    continue
                vals = []
                for src, dst in ((std_a, std_b), (std_b, std_a)):
                    gal = {u: tuple(st.mean(w[i] for w in src[u]) for i in range(3)) for u in users}
                    vals.append(rank1(gal, dst, axes, users, n_gal, draws=20, rng=rng))
                per_n[n_gal] = {"mean": sum(vals) / 2, "a_to_b": vals[0], "b_to_a": vals[1],
                                "chance": 1.0 / n_gal}
            results["groups"][gid]["axes"][name] = per_n
            cells = "  ".join(f"N={n}: {v['mean']:.3f} (chance {v['chance']:.3f})"
                              for n, v in sorted(per_n.items()))
            print(f"   {name:8s} {cells}")

    (Path(__file__).parent / "questset_static_lookup.json").write_text(json.dumps(results, indent=2))

    # Score EVERY registered band, not only the one the falsifier is written on.
    # The first version of this block scored group 2 alone and printed a single
    # verdict, which was then read as the verdict for the whole registration --
    # while group 1's registered level had in fact been missed (2026-09-16).
    g2y = results["groups"]["2"]["axes"]["y only"][30]["mean"]
    g1y = results["groups"]["1"]["axes"]["y only"][30]["mean"]
    g2_verdict = ("FALSIFIER FIRED - withdraw the geometry reading" if g2y > 0.15 else
                  "weakened - reduced but not destroyed" if g2y >= 0.10 else
                  "HELD - at chance as predicted")
    g1_verdict = ("HELD - above the registered 0.15" if g1y > 0.15 else
                  "NOT MET - above chance but below the registered 0.15")
    whole = "HELD" if (g1y > 0.15 and g2y < 0.10) else "PARTIAL - see both lines"
    results["verdict"] = {
        "group_1_y_only_N30": {"value": g1y, "registered": "> 0.15", "verdict": g1_verdict},
        "group_2_y_only_N30": {"value": g2y, "registered": "< 0.10", "verdict": g2_verdict},
        "whole_registration": whole,
    }
    (Path(__file__).parent / "questset_static_lookup.json").write_text(json.dumps(results, indent=2))

    print("\nVERDICT against the registration - BOTH bands, y only, N=30:")
    print(f"  group 1  {g1y:.3f}  registered > 0.15  -> {g1_verdict}")
    print(f"  group 2  {g2y:.3f}  registered < 0.10  -> {g2_verdict}")
    print(f"  whole registration: {whole}")
    print("  (y-only is scale-invariant, so standardised and raw-metre values coincide.)")


if __name__ == "__main__":
    main()

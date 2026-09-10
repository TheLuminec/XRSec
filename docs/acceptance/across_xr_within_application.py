"""Within-application placement on Across-XR: is placement a per-participant constant
when the GAME is held fixed?

WHY THIS EXISTS. The coordinator measured P(within-participant < between-participant) on
per-game mean head position ACROSS applications and got 0.527 laterally - chance - and
concluded a cross-application pair is largely free of the placement artefact. Trainer
pointed out that says nothing about WITHIN-application placement. If placement is a
per-participant constant inside one game, then a same-application control arm carries a
placement cue the cross-application arm does not, and any activity effect measured against
that control is inflated in the flattering direction.

THE CONFOUND THIS DESIGN EXISTS TO AVOID. Two adjacent segments of one recording are
similar because head position drifts slowly, not because placement is a person's constant.
Both shrink the within-participant distance and only one of them is the thing being
measured. So the within-participant comparison must sit at a temporal separation
comparable to the cross-application one, and this script reports TWO splits rather than
one - `take` (different take_id, the same short break that separates the coordinator's
games) and `half` (first vs last half of one take, maximally separated within it). If they
disagree, that gap is a result about drift and gets reported, not averaged away.

DIRECTION IS PINNED BY A FIXTURE. This exact statistic was published inverted once on this
corpus - P(between < within) under the opposite label - and was caught only because the
printed medians contradicted it. `--selftest` runs synthetic cases whose answers are known
by construction (participants 10 m apart must return 1.000; identical participants ~0.5)
and asserts on them before any real file is opened.

Usage:
    python across_xr_within_application.py --selftest
    python across_xr_within_application.py --raw-dir raw_datasets/Across_XR_Dataset_Main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AXES = {"all": (0, 1, 2), "lateral": (0, 2), "height": (1,)}
CENTIMETRES_PER_METRE = 100.0
POSITION_COLUMNS = ["head_pos_x", "head_pos_y", "head_pos_z"]
ID_COLUMNS = ["user_id", "game_id", "take_id"]
TIME_COLUMN = "timestamp"


def p_within_below_between(within: np.ndarray, between: np.ndarray) -> float:
    """P(a random within-participant distance < a random between-participant distance).

    Computed as the mean over within-distances of the fraction of between-distances that
    EXCEED it. The inverted form - `searchsorted(sorted_between, within) / n` - counts
    between-distances BELOW each within-distance, which is P(between < within), and is
    the error this statistic has already produced once on this corpus.
    """
    within = np.asarray(within, dtype=float)
    between = np.asarray(between, dtype=float)
    if within.size == 0 or between.size == 0:
        return float("nan")
    ordered = np.sort(between)
    # count of between STRICTLY GREATER than each within value
    greater = ordered.size - np.searchsorted(ordered, within, side="right")
    ties = np.searchsorted(ordered, within, side="right") - np.searchsorted(ordered, within, side="left")
    return float(np.mean((greater + 0.5 * ties) / ordered.size))


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of `a` and every row of `b`."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).ravel()


def statistic(segments: dict, axis: str) -> dict:
    """P(within < between) with the GAME HELD FIXED on both sides.

    `segments` maps (user, game) -> array of per-segment mean positions, metres.

    Between-participant distances are drawn only from the SAME game. That is the whole
    point: comparing a participant's Beat Saber segments against another participant's
    Alyx segments would fold the activity difference back into the statistic this is
    meant to isolate from it.
    """
    columns = list(AXES[axis])
    within_all, between_all = [], []
    games = sorted({game for _, game in segments})
    for game in games:
        users = sorted({user for user, g in segments if g == game})
        per_user = {u: segments[(u, game)][:, columns] for u in users}
        for user in users:
            block = per_user[user]
            if block.shape[0] >= 2:
                distances = _pairwise(block, block)
                # drop the zero self-distances on the diagonal
                diag = np.eye(block.shape[0], dtype=bool).ravel()
                within_all.append(distances[~diag])
        for i, user_a in enumerate(users):
            for user_b in users[i + 1:]:
                between_all.append(_pairwise(per_user[user_a], per_user[user_b]))
    if not within_all or not between_all:
        return {"p": float("nan"), "within_median": float("nan"),
                "between_median": float("nan"), "n_within": 0, "n_between": 0}
    within = np.concatenate(within_all)
    between = np.concatenate(between_all)
    return {"p": p_within_below_between(within, between),
            "within_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "n_within": int(within.size), "n_between": int(between.size)}


def bootstrap(segments: dict, axis: str, n_boot: int, seed: int) -> tuple:
    """Percentile CI, resampling PARTICIPANTS - the unit of independence is the person,
    not the pair. Bootstrapping pairs would treat 49 people as thousands of observations
    and report an interval several times too tight."""
    rng = np.random.default_rng(seed)
    users = sorted({user for user, _ in segments})
    values = []
    for _ in range(n_boot):
        drawn = rng.choice(users, size=len(users), replace=True)
        resampled = {}
        for new_index, user in enumerate(drawn):
            for (u, game), block in segments.items():
                if u == user:
                    resampled[(f"{new_index}", game)] = block
        value = statistic(resampled, axis)["p"]
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def segments_from_frame(frame, split: str) -> dict:
    """Per-(user, game) arrays of segment mean positions in METRES.

    `take` - one segment per take_id.
    `half` - two segments per take: its first and last half by time, which is the
             maximally separated pair available inside a single take.
    """
    import pandas as pd

    out: dict = {}
    positions = frame[POSITION_COLUMNS].to_numpy(dtype=float) / CENTIMETRES_PER_METRE
    users = frame["user_id"].to_numpy()
    games = frame["game_id"].to_numpy()
    takes = frame["take_id"].to_numpy()
    if TIME_COLUMN in frame.columns:
        try:
            seconds = pd.to_timedelta(frame[TIME_COLUMN]).dt.total_seconds().to_numpy()
        except Exception:
            seconds = np.arange(len(frame), dtype=float)
    else:
        seconds = np.arange(len(frame), dtype=float)

    for user in np.unique(users):
        for game in np.unique(games[users == user]):
            base = (users == user) & (games == game)
            blocks = []
            for take in np.unique(takes[base]):
                mask = base & (takes == take)
                if mask.sum() < 2:
                    continue
                if split == "take":
                    blocks.append(positions[mask].mean(axis=0))
                else:
                    order = np.argsort(seconds[mask], kind="stable")
                    rows = positions[mask][order]
                    cut = len(rows) // 2
                    if cut < 1 or len(rows) - cut < 1:
                        continue
                    blocks.append(rows[:cut].mean(axis=0))
                    blocks.append(rows[cut:].mean(axis=0))
            if len(blocks) >= 1:
                out[(str(user), int(game))] = np.asarray(blocks, dtype=float)
    return out


def load_raw(raw_dir: Path):
    import pandas as pd

    frames = []
    files = sorted(raw_dir.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
    files = [f for f in files if f.stem.isdigit()]
    if not files:
        raise SystemExit(f"no <N>.csv files under {raw_dir}")
    print(f"reading {len(files)} files from {raw_dir}")
    for path in files:
        frame = pd.read_csv(path, usecols=lambda c: c in set(POSITION_COLUMNS + ID_COLUMNS + [TIME_COLUMN]))
        missing = [c for c in POSITION_COLUMNS + ID_COLUMNS if c not in frame.columns]
        if missing:
            raise SystemExit(f"{path.name} missing columns: {missing}")
        frames.append(frame)
        print(f"  {path.name}: {len(frame):,} rows", flush=True)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- self-test

def _synthetic(separation_m: float, jitter_m: float, n_users=12, n_games=3, n_seg=4, seed=0):
    rng = np.random.default_rng(seed)
    segments = {}
    for user in range(n_users):
        centre = rng.normal(0, separation_m, size=3)
        for game in range(n_games):
            segments[(str(user), game)] = centre + rng.normal(0, jitter_m, size=(n_seg, 3))
    return segments


def selftest() -> int:
    print("FIXTURE 1: participants 10 m apart, segments jittered by 1 cm")
    far = _synthetic(separation_m=10.0, jitter_m=0.01, seed=1)
    assert len(far) == 36, f"fixture failed to build: {len(far)} cells, expected 36"
    sample = next(iter(far.values()))
    assert sample.shape == (4, 3), f"fixture segment block is {sample.shape}, expected (4, 3)"
    for axis in AXES:
        result = statistic(far, axis)
        print(f"  {axis:8} P={result['p']:.4f}  within_med={result['within_median']:.4f} "
              f"between_med={result['between_median']:.4f}")
        assert result["p"] > 0.99, f"{axis}: separated participants must give P~1, got {result['p']}"
        assert result["within_median"] < result["between_median"], \
            f"{axis}: medians contradict P - the exact inconsistency that caught the inverted version"

    print("FIXTURE 2: participants drawn from ONE distribution (no per-person constant)")
    same = _synthetic(separation_m=1e-9, jitter_m=1.0, seed=2)
    for axis in AXES:
        result = statistic(same, axis)
        print(f"  {axis:8} P={result['p']:.4f}")
        assert 0.40 < result["p"] < 0.60, f"{axis}: indistinguishable participants must give P~0.5, got {result['p']}"

    print("FIXTURE 3: direction - swapping the arguments must give 1-P, not P")
    within = np.array([0.1, 0.2, 0.3])
    between = np.array([0.15, 0.25, 0.35, 0.45])
    forward = p_within_below_between(within, between)
    backward = p_within_below_between(between, within)
    print(f"  P(within<between)={forward:.4f}  P(between<within)={backward:.4f}  sum={forward + backward:.4f}")
    assert abs(forward + backward - 1.0) < 1e-9, "the statistic is not antisymmetric - direction is wrong"
    assert forward > backward, "with within shifted below between, P must exceed 0.5"

    print("FIXTURE 4: the INVERTED implementation must fail fixture 1 (test is not vacuous)")
    far_stat = statistic(far, "lateral")["p"]
    # the published bug: searchsorted without the complement
    b = np.sort(np.concatenate([_pairwise(far[("0", 0)], far[("1", 0)])]))
    w = _pairwise(far[("0", 0)], far[("0", 0)])
    w = w[w > 0]
    bugged = float(np.mean(np.searchsorted(b, w, side="right") / b.size))
    print(f"  correct={far_stat:.4f}  bugged(P(between<within))={bugged:.4f}")
    assert bugged < 0.5 < far_stat, "the bugged form should read LOW where the correct one reads high"

    print("\nALL FIXTURES PASS - direction pinned, medians consistent, test non-vacuous.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, help="directory of Across-XR <N>.csv files")
    parser.add_argument("--selftest", action="store_true", help="run fixtures and exit")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/acceptance/across_xr_within_application.json"))
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.raw_dir:
        parser.error("--raw-dir is required unless --selftest")

    # The fixtures gate the real run: a harness that cannot pass them must not produce a number.
    print("=== gating on fixtures before touching real data ===")
    selftest()
    print()

    frame = load_raw(args.raw_dir)
    print(f"\ntotal rows: {len(frame):,}")
    report = {"registered": {"lateral": [0.80, 0.95], "height": [0.85, 0.97], "falsifier_lateral_below": 0.65},
              "splits": {}}

    for split in ("take", "half"):
        segments = segments_from_frame(frame, split)
        users = sorted({u for u, _ in segments})
        cells = len(segments)
        counts = [v.shape[0] for v in segments.values()]
        print(f"\n--- split={split}: {len(users)} participants, {cells} (user,game) cells, "
              f"segments/cell min={min(counts)} median={int(np.median(counts))} max={max(counts)} ---")
        if max(counts) < 2:
            print(f"    SKIPPED: no (user,game) cell has 2+ segments under split={split}")
            report["splits"][split] = {"skipped": "no cell has 2+ segments"}
            continue
        entry = {"n_participants": len(users), "n_cells": cells}
        for axis in ("all", "lateral", "height"):
            result = statistic(segments, axis)
            low, high = bootstrap(segments, axis, args.n_boot, args.seed)
            entry[axis] = {**result, "ci95": [low, high]}
            print(f"  {axis:8} P={result['p']:.4f}  CI95=[{low:.4f}, {high:.4f}]  "
                  f"within_med={result['within_median']:.4f} m  between_med={result['between_median']:.4f} m")
            if result["within_median"] > result["between_median"] and result["p"] > 0.5:
                print("    *** WARNING: medians and P disagree - check direction before quoting ***")
        report["splits"][split] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")

    lateral = report["splits"].get("take", {}).get("lateral") or report["splits"].get("half", {}).get("lateral")
    if lateral:
        low, high = lateral["ci95"]
        print("\n=== registered band, read against the INTERVAL ===")
        print(f"  lateral P = {lateral['p']:.4f}, CI95 [{low:.4f}, {high:.4f}]")
        print(f"  band 0.80-0.95: {'CONTAINED' if 0.80 <= low and high <= 0.95 else 'not contained'}")
        print(f"  falsifier <0.65: {'FIRED' if high < 0.65 else 'did not fire'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

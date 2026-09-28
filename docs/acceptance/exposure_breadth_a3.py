"""
Exposure breadth, Amendment 4: the Beat Saber (covered/uncovered) split on Questset, from committed
files only. Breadth - treatment per user at N=30, per group; group 2's titles are in no training
corpus, group 1 contains Beat Saber, which four of five breadth checkpoints saw in Across-XR.

    python docs/acceptance/exposure_breadth_a3.py --seeds 1 [2 3] [--out ...]

Seed s reads exposure_breadth_questset_<X>[_s<s>].json (no suffix for seed 1) and
exposure_breadth_questset_treatment_s<s>_gpu.json. Per user: breadth averaged over the five X,
minus the seed-matched treatment, then averaged over seeds; user bootstrap per group.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exposure_breadth_read import ci  # noqa: E402  one bootstrap implementation
from across_xr_alignment_p3 import GAMES  # noqa: E402

def verdict_gain(r):
    if r["lo"] > 0:
        return "BAND: whole interval > 0, exposure carries to titles no corpus covers"
    if r["hi"] <= 0:
        return "FALSIFIER: whole interval <= 0, the Questset gain is coverage"
    return "interval straddles 0: not resolved"


def verdict_split(r):
    if r["hi"] <= 0.05:
        return "holds: interval upper <= +0.05, coverage adds at most +0.05"
    if r["lo"] > 0.05:
        return "FALSIFIER: interval lower > +0.05, coverage drives it"
    return "interval straddles +0.05: not resolved"


def per_user(path, group):
    d = json.loads(pathlib.Path(path).read_text())
    assert d["gate"]["gap"] < 1e-4, (path, d["gate"]["gap"])
    cell = d["groups"][group]["cells"]["30"]
    assert "per_user" in cell, f"{path}: no per-user N=30 record for group {group}"
    return cell["per_user"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(67)
    out = {"seeds": a.seeds, "groups": {}, "verdicts": {}, "descriptive_beat_saber_heldout": {}}
    diffs = {}
    for g in ("1", "2"):
        per_seed = []
        for s in a.seeds:
            suffix = "" if s == 1 else f"_s{s}"
            b = [per_user(HERE / f"exposure_breadth_questset_{x}{suffix}.json", g) for x in GAMES]
            t = per_user(HERE / f"exposure_breadth_questset_treatment_s{s}_gpu.json", g)
            users = sorted(t)
            assert all(sorted(x) == users for x in b) and len(users) == 30, g
            per_seed.append(np.mean([[x[u] for u in users] for x in b], axis=0) - np.array([t[u] for u in users]))
            if g == "1":
                bs = np.array([b[GAMES.index("beat_saber")][u] for u in users]) - np.array([t[u] for u in users])
                rest = np.mean([[b[i][u] for u in users] for i, x in enumerate(GAMES) if x != "beat_saber"], axis=0) \
                    - np.array([t[u] for u in users])
                out["descriptive_beat_saber_heldout"][f"seed{s}"] = {
                    "beat_saber_heldout_gain": float(bs.mean()), "other_four_gain": float(rest.mean())}
        diffs[g] = np.mean(per_seed, axis=0)
        out["groups"][g] = ci(diffs[g], rng)
    # group1 - group2 over DIFFERENT people: independent bootstrap of each group, difference of means
    draws = 10000
    g1, g2 = diffs["1"], diffs["2"]
    b1 = g1[rng.integers(0, len(g1), (draws, len(g1)))].mean(axis=1)
    b2 = g2[rng.integers(0, len(g2), (draws, len(g2)))].mean(axis=1)
    dd = b1 - b2
    out["groups"]["1-2"] = {"mean": float(g1.mean() - g2.mean()), "lo": float(np.percentile(dd, 2.5)),
                            "hi": float(np.percentile(dd, 97.5))}
    out["verdicts"]["group2_gain"] = verdict_gain(out["groups"]["2"])
    out["verdicts"]["group1_minus_group2"] = verdict_split(out["groups"]["1-2"])
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")
    for k in ("1", "2", "1-2"):
        r = out["groups"][k]
        print(f"  breadth - treatment, group {k}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v}")
    for s, v in out["descriptive_beat_saber_heldout"].items():
        print(f"  {s} group 1, descriptive: beat_saber-held-out gain {v['beat_saber_heldout_gain']:+.4f}, other four {v['other_four_gain']:+.4f}")


if __name__ == "__main__":
    main()

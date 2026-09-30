"""
Read ballthrowing_cross_day results against ballthrowing_cross_day_REGISTERED.md. Committed before any
result existed. Per user: each condition averaged over the gated seeds, then a user bootstrap (the
exposure-breadth `ci`, one implementation).

    python docs/acceptance/ballthrowing_cross_day_read.py <result json> [...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exposure_breadth_read import ci  # noqa: E402


def v_level(r):
    if r["lo"] >= 0.10:
        return "BAND: identifies across days from 2 s throws (>= 4x chance)"
    if r["hi"] < 0.05:
        return "FALSIFIER: no cross-day identification at this window"
    return "weak: the interval sits in or spans the 0.05-0.10 region"


def v_day(r):
    if r["lo"] >= -0.15:
        return "BAND: modest day cost"
    if r["hi"] < -0.30:
        return "FALSIFIER: most same-session identification is session-specific"
    if r["lo"] >= -0.30 and r["hi"] < -0.15:
        return "partial (-0.30..-0.15)"
    return f"spans regions; mean {'modest' if r['mean'] >= -0.15 else 'partial' if r['mean'] >= -0.30 else 'falsifier'}"


def v_headset(r):
    if r["lo"] >= -0.10:
        return "BAND: a headset change costs little beyond the day"
    if r["hi"] < -0.20:
        return "FALSIFIER: identity does not survive a headset change"
    if r["lo"] >= -0.20 and r["hi"] < -0.10:
        return "substantial (-0.20..-0.10)"
    return f"spans regions; mean {'little' if r['mean'] >= -0.10 else 'substantial' if r['mean'] >= -0.20 else 'falsifier'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(67)
    res = [r for f in a.files for r in json.loads(pathlib.Path(f).read_text())["results"]]
    ok = [r for r in res if "refused" not in r]
    refused = [(r["seed"], r["refused"]) for r in res if "refused" in r]
    assert ok, "no gated result to read"
    users = sorted(ok[0]["C0"])
    assert all(sorted(r["C0"]) == users for r in ok) and len(users) == 41
    m = {k: np.array([np.mean([r[k][u] for r in ok]) for u in users])
         for k in ("C0", "C1", "C2", "C1_height", "C2_height")}
    le3 = np.array([np.nanmean([r["C2_le3d"][u] for r in ok]) for u in users])
    has = ~np.isnan(le3)
    out = {"seeds": sorted(r["seed"] for r in ok), "refused": refused, "chance": 1 / 41,
           "levels": {k: ci(v, rng) for k, v in m.items()},
           "C1_level": ci(m["C1"], rng), "day_cost_C1-C0": ci(m["C1"] - m["C0"], rng),
           "headset_cost_C2-C1": ci(m["C2"] - m["C1"], rng),
           "gap_matched_C2le3d-C1_descriptive": {**ci(le3[has] - m["C1"][has], rng), "n_users": int(has.sum())}}
    out["verdicts"] = {"C1_level": v_level(out["C1_level"]), "day_cost": v_day(out["day_cost_C1-C0"]),
                       "headset_cost": v_headset(out["headset_cost_C2-C1"])}
    for k in ("C1_level", "day_cost_C1-C0", "headset_cost_C2-C1", "gap_matched_C2le3d-C1_descriptive"):
        r = out[k]; print(f"  {k:36s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    for k, r in out["levels"].items():
        print(f"  level {k:10s} {r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]")
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v}")
    for s in refused:
        print("  refused (not read):", s)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

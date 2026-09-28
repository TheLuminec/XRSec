"""
Read alyx_cross_day results against alyx_cross_day_REGISTERED.md. Treatment: (seed, user) units pooled
over seeds; breadth: per seed, the gated checkpoints averaged per user; breadth - treatment seed-paired
and descriptive. Refused checkpoints are excluded and listed.

    python docs/acceptance/alyx_cross_day_read.py docs/acceptance/alyx_cross_day_s1.json [...] --out ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exposure_breadth_read import ci  # noqa: E402


def level_verdict(r):
    if r["lo"] >= 0.20:
        return "BAND: the learned component identifies people across days"
    if r["hi"] < 0.10:
        return "FALSIFIER: no cross-day identification beyond ~1.5x chance"
    if r["lo"] >= 0.10 and r["hi"] < 0.20:
        return "weak (0.10-0.20)"
    return "interval spans regions: " + ("mean in band" if r["mean"] >= 0.20 else "mean weak" if r["mean"] >= 0.10 else "mean below 0.10")


def cost_verdict(r):
    if r["lo"] >= -0.15:
        return "BAND: one-sitting results carry across days at a modest cost"
    if r["hi"] < -0.30:
        return "FALSIFIER: a large share of same-day identification is session-specific"
    if r["lo"] >= -0.30 and r["hi"] < -0.15:
        return "partial: a substantial cost (-0.30..-0.15)"
    mean_region = ("band" if r["mean"] >= -0.15 else "partial" if r["mean"] >= -0.30 else "falsifier")
    return f"interval spans regions (lo {r['lo']:+.3f}, hi {r['hi']:+.3f}); the mean sits in the {mean_region} region"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(67)
    res = [r for f in a.files for r in json.loads(pathlib.Path(f).read_text())["results"]]
    refused = [(r["arm"], r.get("seed"), r["refused"]) for r in res if "refused" in r]
    ok = [r for r in res if "refused" not in r]
    treat = defaultdict(dict)
    breadth = defaultdict(lambda: defaultdict(list))
    for r in ok:
        for u in r["cross_day"]:
            unit = (r["cross_day"][u], r["same_day"][u])
            if r["arm"] == "treatment":
                treat[r["seed"]][u] = unit
            elif r["arm"].startswith("breadth_"):
                breadth[r["seed"]][u].append(unit)
    tc = np.array([v[0] for s in sorted(treat) for v in treat[s].values()])
    ts = np.array([v[1] for s in sorted(treat) for v in treat[s].values()])
    out = {"refused": refused, "treatment": {"seeds": sorted(treat), "units": int(len(tc)),
                                               "cross_day": ci(tc, rng), "same_day": ci(ts, rng), "cost": ci(tc - ts, rng)}}
    out["treatment"]["verdict_level"] = level_verdict(out["treatment"]["cross_day"])
    out["treatment"]["verdict_cost"] = cost_verdict(out["treatment"]["cost"])
    out["breadth_minus_treatment_descriptive"] = {}
    for s, users in breadth.items():
        n_ck = {len(v) for v in users.values()}
        bc = np.array([np.mean([x[0] for x in users[u]]) for u in sorted(users)])
        bs = np.array([np.mean([x[1] for x in users[u]]) for u in sorted(users)])
        tcs = np.array([treat[s][u][0] for u in sorted(users)])
        out["breadth_minus_treatment_descriptive"][f"seed{s}"] = {
            "checkpoints_per_user": sorted(n_ck), "breadth_cross_day": float(bc.mean()), "breadth_same_day": float(bs.mean()),
            "breadth_cost": ci(bc - bs, rng), "cross_day_breadth_minus_treatment": ci(bc - tcs, rng)}
    t = out["treatment"]
    print(f"treatment, seeds {t['seeds']}, {t['units']} (seed,user) units")
    for k in ("cross_day", "same_day", "cost"):
        print(f"  {k:9s} {t[k]['mean']:+.4f} [{t[k]['lo']:+.4f}, {t[k]['hi']:+.4f}]")
    print(f"  level: {t['verdict_level']}\n  cost:  {t['verdict_cost']}")
    for s, d in out["breadth_minus_treatment_descriptive"].items():
        print(f"breadth {s} ({d['checkpoints_per_user']} checkpoints): cross {d['breadth_cross_day']:.3f} same {d['breadth_same_day']:.3f} "
              f"cost {d['breadth_cost']['mean']:+.3f} [{d['breadth_cost']['lo']:+.3f}, {d['breadth_cost']['hi']:+.3f}]; "
              f"cross-day vs treatment {d['cross_day_breadth_minus_treatment']['mean']:+.3f} "
              f"[{d['cross_day_breadth_minus_treatment']['lo']:+.3f}, {d['cross_day_breadth_minus_treatment']['hi']:+.3f}]")
    for x in refused:
        print("refused (not read):", x)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

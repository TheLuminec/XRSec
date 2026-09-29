"""
Exposure breadth, Amendment 4: the three-seed reading. Imports the regions, verdict, bootstrap and
Questset loaders of exposure_breadth_read.py (one implementation) and adds only what Amendment 4
registered: seed aggregation and the hold-out-cost row.

Aggregation (registered): per X, per-user values averaged over the three seeds; P3 seed 1 for every X;
treatment seed-paired (breadth seed s - treatment seed s, then averaged over seeds; s1 treatment scored on
GPU, s2/s3 on AVALON CPU, gates recorded); C2-lo three-seed mean; Questset breadth = all 15 checkpoints,
zero-shot = its three GPU seeds.

    python docs/acceptance/exposure_breadth_read3.py [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exposure_breadth_read as R  # noqa: E402
from across_xr_alignment_aggregate import load  # noqa: E402
from across_xr_alignment_p3 import GAMES, cells_involving, cells_not_involving, mean_over  # noqa: E402

A = HERE
SEEDS = (1, 2, 3)
R.ROWS["holdout_cost_breadth-C2lo_x"] = [
    (-np.inf, -0.05, "FALSIFIER: holding X out does cost at scale; one seed hid it"),
    (-0.05, -0.03, "a small cost"),
    (-0.03, 0.03, "BAND: at scale, four applications substitute for the fifth"),
    (0.03, np.inf, "above +0.03: breadth beats the arm that saw X; read composition (Nymeria) first")]


def breadth_file(x, s):
    return A / (f"exposure_breadth_axr_{x}.json" if s == 1 else f"exposure_breadth_axr_{x}_s{s}.json")


def treatment_file(s):
    return A / ("exposure_breadth_axr_treatment_s1.json" if s == 1 else f"exposure_breadth_axr_treatment_s{s}_cpu.json")


def gated(path):
    seed = load([str(path)])[0]
    assert seed["gate"]["passed"], (path, seed["gate"])
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A / "exposure_breadth_read3.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(67)
    treat = {s: gated(treatment_file(s)) for s in SEEDS}
    c2 = [gated(A / f"across_xr_alignment_c2lo_seed{i}.json") for i in (1, 2, 3)]   # always C2-lo's three seeds, independent of SEEDS
    out = {"per_x": {}, "pooled": {}, "verdicts": {}, "gates": {}}
    rows = {k: [] for k in ("breadth-P3_x", "breadth-treatment_x", "control_breadth-C2lo_nonx", "holdout_cost_breadth-C2lo_x")}
    for x in GAMES:
        b = {s: gated(breadth_file(x, s)) for s in SEEDS}
        out["gates"][x] = {s: b[s]["gate"]["gap"] for s in SEEDS}
        p = gated(A / f"across_xr_alignment_p3_{x}.json")
        xc, nx = cells_involving(x), cells_not_involving(x)
        bx = np.mean([mean_over(b[s], "A1", xc) for s in SEEDS], axis=0)
        bn = np.mean([mean_over(b[s], "A1", nx) for s in SEEDS], axis=0)
        px = mean_over(p, "A1", xc)
        bt = np.mean([mean_over(b[s], "A1", xc) - mean_over(treat[s], "A1", xc) for s in SEEDS], axis=0)
        cx = np.mean([mean_over(c, "A1", xc) for c in c2], axis=0)
        cn = np.mean([mean_over(c, "A1", nx) for c in c2], axis=0)
        per_seed = [float(mean_over(b[s], "A1", xc).mean()) for s in SEEDS]
        out["per_x"][x] = {"breadth_x_per_seed": per_seed, "breadth_x": float(bx.mean()), "P3_x": float(px.mean()),
                           "C2lo_x": float(cx.mean())}
        for k, v in (("breadth-P3_x", bx - px), ("breadth-treatment_x", bt), ("control_breadth-C2lo_nonx", bn - cn),
                     ("holdout_cost_breadth-C2lo_x", bx - cx)):
            rows[k].append(v)
            out["per_x"][x][k] = R.ci(v, rng)
        print(f"  {x:15s} breadth X-cells per seed {', '.join(f'{v:.3f}' for v in per_seed)} | P3 {px.mean():.3f}  C2-lo {cx.mean():.3f}")
    for k, arr in rows.items():
        r = {**R.ci(np.mean(arr, axis=0), rng), "n_x": len(arr)}
        out["pooled"][k] = r
        out["verdicts"][k] = R.verdict(k, r)
    qb = [A / (f"exposure_breadth_questset_{x}.json" if s == 1 else f"exposure_breadth_questset_{x}_s{s}.json")
          for x in GAMES for s in SEEDS]
    qz = [A / f"exposure_breadth_questset_zeroshot_2026-09-10_{r}_train_gpu.json" for r in ("17-24-33", "18-41-34", "20-05-56")]
    ub, b = R.questset_per_user(qb)
    uz, z = R.questset_per_user(qz)
    assert ub == uz
    r = {**R.ci(b - z, rng), "n_users": len(ub), "breadth_mean": float(b.mean()), "zeroshot_mean": float(z.mean()),
         "n_breadth_checkpoints": len(qb)}
    out["pooled"]["questset_breadth-zeroshot"] = r
    out["verdicts"]["questset_breadth-zeroshot"] = R.verdict("questset_breadth-zeroshot", r)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=float))       # artefact first
    print()
    for k, r in out["pooled"].items():
        print(f"  {k:30s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {out['verdicts'][k]}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

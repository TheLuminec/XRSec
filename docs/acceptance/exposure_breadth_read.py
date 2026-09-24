"""
Experiment 2 (exposure breadth) - the reading, written 2026-09-24 BEFORE any X-cell or Questset
number for the breadth arm existed. It reads committed files only and re-embeds nothing.

    python docs/acceptance/exposure_breadth_read.py            # reads the committed files
    python docs/acceptance/exposure_breadth_read.py --selftest # stand-in inputs, plumbing only

Registered rows (exposure_breadth_REGISTERED.md), each scored by WHERE THE INTERVAL FALLS, never by
p < 0.05. Unit: rank-1, per-user differences on the same 17 Across-XR users (32-48), averaged over
cells, then over the five held-out applications X, then a user bootstrap (the P3 aggregation,
imported so there is one implementation).

  1. breadth-X - P3-X on the eight X-cells.
  2. breadth-X - Nymeria treatment s1 on the same X-cells.
  3. control: breadth-X - C2-lo on the twelve non-X cells. C2-lo is the three-seed mean per user
     per cell (fixed here, before reading).
  4. Questset cross-game rank-1: breadth (mean of the five X checkpoints) - zero-shot (mean of the
     three seeds), per user at N=30 (the whole group is the gallery), pooled over both groups
     (60 users), user bootstrap. N=17 group means are reported beside it, without an interval.

Regions the registration left unnamed are named HERE, before reading (the band-partition rule):
  row 1 above +0.06        -> "above band: breadth adds more than registered; report as exceeding"
  row 4 -0.03 .. 0.00      -> "not resolved"
  row 4 above +0.10        -> "above +0.10: breadth transfers to a fully unseen corpus; seed it"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from across_xr_alignment_aggregate import boot, load  # noqa: E402
from across_xr_alignment_p3 import GAMES, cells_involving, cells_not_involving, mean_over  # noqa: E402

# (lower edge, upper edge, meaning) - contiguous, covering the whole line
ROWS = {
    "breadth-P3_x": [(-np.inf, -0.03, "FALSIFIER: more and broader data hurts the held-out application"),
                     (-0.03, 0.00, "not resolved: run the balance_identities=cap companion before reading"),
                     (0.00, 0.06, "BAND: scale and daily-life tasks add a little to exposure"),
                     (0.06, np.inf, "above band: breadth adds more than registered; report as exceeding")],
    "breadth-treatment_x": [(-np.inf, 0.00, "FALSIFIER: with Nymeria in training, Across-XR exposure adds nothing"),
                            (0.00, 0.03, "unresolved at one seed"),
                            (0.03, np.inf, "BAND: exposure to four applications carries on top of Nymeria breadth")],
    "control_breadth-C2lo_nonx": [(-np.inf, -0.03, "outside: removing one application moved the seen ones"),
                                  (-0.03, 0.03, "within +-0.03: the control holds"),
                                  (0.03, np.inf, "outside: removing one application moved the seen ones")],
    "questset_breadth-zeroshot": [(-np.inf, -0.03, "FALSIFIER"),
                                  (-0.03, 0.00, "not resolved"),
                                  (0.00, 0.05, "BAND"),
                                  (0.05, 0.10, "breadth transfers to a fully unseen corpus: seed it"),
                                  (0.10, np.inf, "above +0.10: breadth transfers to a fully unseen corpus; seed it")],
}


def regions_hit(row, lo, hi):
    return [meaning for a, b, meaning in ROWS[row] if lo < b and hi >= a]


def verdict(row, r):
    hit = regions_hit(row, r["lo"], r["hi"])
    at_mean = regions_hit(row, r["mean"], r["mean"])
    if len(hit) == 1:
        return f"interval inside one region: {hit[0]}"
    return f"interval spans {len(hit)} regions ({' | '.join(hit)}); the mean sits in: {at_mean[0]}"


def ci(values, rng):
    m, lo, hi = boot(np.asarray(values, dtype=float), rng)
    return {"mean": float(m), "lo": float(lo), "hi": float(hi)}


def questset_per_user(paths):
    """Mean over checkpoints of per-user rank-1 at N=30, keyed by user; asserts identical user sets."""
    per = []
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        assert d["gate"]["gap"] < 1e-4, (p, d["gate"]["gap"])
        pu = {}
        for gid, g in d["groups"].items():
            cell = g["cells"].get("30") or g["cells"].get(30)
            assert cell and "per_user" in cell, f"{p}: group {gid} has no per-user N=30 record"
            pu.update(cell["per_user"])
        per.append(pu)
    users = sorted(per[0])
    assert all(sorted(p) == users for p in per) and len(users) == 60, "Questset user sets differ"
    return users, np.mean([[p[u] for u in users] for p in per], axis=0)


def questset_n17(paths):
    out = {}
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        for gid, g in d["groups"].items():
            cell = g["cells"].get("17") or g["cells"].get(17)
            out.setdefault(gid, []).append(cell["mean"])
    return {gid: float(np.mean(v)) for gid, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breadth", nargs="+", default=[f"{x}=docs/acceptance/exposure_breadth_axr_{x}.json" for x in GAMES])
    ap.add_argument("--p3", nargs="+", default=[f"{x}=docs/acceptance/across_xr_alignment_p3_{x}.json" for x in GAMES])
    ap.add_argument("--treatment", default="docs/acceptance/exposure_breadth_axr_treatment_s1.json")
    ap.add_argument("--c2lo", nargs="+", default=[f"docs/acceptance/across_xr_alignment_c2lo_seed{i}.json" for i in (1, 2, 3)])
    ap.add_argument("--qs-breadth", nargs="+", default=[f"docs/acceptance/exposure_breadth_questset_{x}.json" for x in GAMES])
    ap.add_argument("--qs-zeroshot", nargs="+", default=[f"docs/acceptance/exposure_breadth_questset_zeroshot_{r}_gpu.json"
                                                          for r in ("17-24-33", "18-41-34", "20-05-56")])
    ap.add_argument("--out", default=str(HERE / "exposure_breadth_read.json"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-questset", action="store_true",
                    help="rows 1-3 only (plumbing switch; the analysis of each row is unchanged)")
    args = ap.parse_args()
    rng = np.random.default_rng(67)

    as_map = lambda items: {k: v for k, v in (i.split("=", 1) for i in items)}  # noqa: E731
    breadth, p3 = as_map(args.breadth), as_map(args.p3)
    assert sorted(breadth) == sorted(p3) == sorted(GAMES)
    t = load([args.treatment])[0]
    c2 = [load([p])[0] for p in args.c2lo]
    out = {"per_x": {}, "pooled": {}, "verdicts": {}}
    d1, d2, d3 = [], [], []
    for x in GAMES:
        b, p = load([breadth[x]])[0], load([p3[x]])[0]
        # the gate this reading rests on, per checkpoint
        assert b["gate"]["passed"], (x, b["gate"])
        xc, nx = cells_involving(x), cells_not_involving(x)
        assert len(xc) == 8 and len(nx) == 12
        bx, px, tx = mean_over(b, "A1", xc), mean_over(p, "A1", xc), mean_over(t, "A1", xc)
        bn = mean_over(b, "A1", nx)
        cn = np.mean([mean_over(c, "A1", nx) for c in c2], axis=0)
        rec = {"breadth_x": float(bx.mean()), "P3_x": float(px.mean()), "treatment_x": float(tx.mean()),
               "breadth-P3_x": ci(bx - px, rng), "breadth-treatment_x": ci(bx - tx, rng),
               "control_breadth-C2lo_nonx": ci(bn - cn, rng), "gate": b["gate"]}
        out["per_x"][x] = rec
        d1.append(bx - px); d2.append(bx - tx); d3.append(bn - cn)
        print(f"  {x:15s} X-cells breadth {bx.mean():.3f}  P3 {px.mean():.3f}  treatment {tx.mean():.3f} | "
              f"b-P3 {rec['breadth-P3_x']['mean']:+.3f}  b-T {rec['breadth-treatment_x']['mean']:+.3f}  "
              f"non-X b-C2lo {rec['control_breadth-C2lo_nonx']['mean']:+.3f}")
    for row, arr in (("breadth-P3_x", d1), ("breadth-treatment_x", d2), ("control_breadth-C2lo_nonx", d3)):
        r = {**ci(np.mean(arr, axis=0), rng), "n_x": len(arr)}
        out["pooled"][row] = r
        out["verdicts"][row] = verdict(row, r)

    if not args.skip_questset:
        qs_users, qb = questset_per_user(args.qs_breadth)
        qz_users, qz = questset_per_user(args.qs_zeroshot)
        assert qs_users == qz_users
        r = {**ci(qb - qz, rng), "n_users": len(qs_users), "breadth_mean": float(qb.mean()), "zeroshot_mean": float(qz.mean())}
        out["pooled"]["questset_breadth-zeroshot"] = r
        out["verdicts"]["questset_breadth-zeroshot"] = verdict("questset_breadth-zeroshot", r)
        out["questset_n17_group_means"] = {"breadth": questset_n17(args.qs_breadth), "zeroshot": questset_n17(args.qs_zeroshot)}

    print()
    for row, r in out["pooled"].items():
        print(f"  {row:28s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {out['verdicts'][row]}")
    if args.selftest:
        print("\nSELFTEST: stand-in inputs; these figures are plumbing, not results. Nothing written.")
        return 0
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

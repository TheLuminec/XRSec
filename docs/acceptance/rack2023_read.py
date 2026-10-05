"""
Read the Rack et al. 2023 re-run against sota_rack2023_reproduction_REGISTERED.md (top-of-file gate, unchanged
by Amendments 1-8). Committed BEFORE any test-split number was read.

Registered rule. PASS requires all three:
  1. Each cell: the published value falls inside our measured seed spread, "mean +- spread over 3 seeds".
  2. Ordering: all-enrolment/5 min > 10 min/5 min > 1 min/1 min (on the three-seed means).
  3. Dynamic range: (all/5 - 1/1) lies within the measured spread of the published 74-point gap.
  The 1-min/1-min cell is primary. If any cell falls outside, or the ordering breaks, the reproduction FAILS.

Two instrument facts fixed here before reading, neither chosen from a number:
  - "spread" is the seed RANGE (max - min), this project's usage ("seed range 0.010"), so the interval is
    mean +- range. The two narrower readings, [min, max] and mean +- sd, are reported beside it and are not
    the verdict.
  - The published figures are whole percentages (99 / 89 / 25), so each is the interval [p - 0.005, p + 0.005].
    A cell is "inside" when that interval overlaps ours. The 74-point gap is a difference of two rounded
    figures, so it is [0.73, 0.75].

    python docs/acceptance/rack2023_read.py [--selftest] [--out docs/acceptance/rack2023_read.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

A = pathlib.Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
CELLS = {"all enrolment / 5 min": ("all", "sequence_top_1_accuracy_5_mins", 0.99),
         "10 min / 5 min": ("10.0", "sequence_top_1_accuracy_5_mins", 0.89),
         "1 min / 1 min": ("1.0", "sequence_top_1_accuracy_1_mins", 0.25)}
PUB_HALF = 0.005
GAP_PUB = (0.73, 0.75)


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def read(docs):
    for d, s in zip(docs, SEEDS):
        assert d["seed"] == s and d["split"] == "test" and d["n_test_subjects"] == 27, (d["seed"], d["split"])
    subj = [tuple(d["by_enrolment"]["all"]["subject_ids"]) for d in docs]
    assert len(set(subj)) == 1, "the three seeds were not scored on one test set"
    out = {"seeds": list(SEEDS), "n_test_subjects": 27, "cells": {}, "verdicts": {}}
    means = {}
    for name, (enrol, metric, pub) in CELLS.items():
        v = np.array([d["by_enrolment"][enrol]["metrics"][metric] for d in docs], dtype=float)
        m, rng, sd = v.mean(), v.max() - v.min(), v.std(ddof=1)
        pub_iv = (pub - PUB_HALF, pub + PUB_HALF)
        out["cells"][name] = {"per_seed": v.tolist(), "mean": m, "range": rng, "sd": sd, "published": pub,
                              "inside_mean_pm_range": overlaps(pub_iv, (m - rng, m + rng)),
                              "inside_min_max": overlaps(pub_iv, (v.min(), v.max())),
                              "inside_mean_pm_sd": overlaps(pub_iv, (m - sd, m + sd))}
        means[name] = m
    names = list(CELLS)
    gaps = np.array(out["cells"][names[0]]["per_seed"]) - np.array(out["cells"][names[2]]["per_seed"])
    g_m, g_r = gaps.mean(), gaps.max() - gaps.min()
    out["dynamic_range"] = {"per_seed": gaps.tolist(), "mean": g_m, "range": g_r, "published": GAP_PUB,
                            "inside_mean_pm_range": overlaps(GAP_PUB, (g_m - g_r, g_m + g_r))}
    ordering = means[names[0]] > means[names[1]] > means[names[2]]
    cells_ok = all(c["inside_mean_pm_range"] for c in out["cells"].values())
    out["verdicts"] = {
        "each_cell": "PASS" if cells_ok else "FAIL: " + ", ".join(n for n, c in out["cells"].items() if not c["inside_mean_pm_range"]),
        "primary_1min_1min": "inside" if out["cells"][names[2]]["inside_mean_pm_range"] else "OUTSIDE",
        "ordering": "PASS" if ordering else "FAIL",
        "dynamic_range": "PASS" if out["dynamic_range"]["inside_mean_pm_range"] else "FAIL",
    }
    out["verdicts"]["overall"] = ("PASS: the reproduction of the published curve holds"
                                  if cells_ok and ordering and out["dynamic_range"]["inside_mean_pm_range"]
                                  else "FAIL: the reproduction does not hold under the registered rule")
    out["table"] = {e: {m: [d["by_enrolment"][e]["metrics"][m] for d in docs]
                        for m in ("sequence_top_1_accuracy_1_mins", "sequence_top_1_accuracy_5_mins")}
                    for e in docs[0]["by_enrolment"]}
    return out


def selftest():
    def fake(vals):        # vals: {enrol: (1min, 5min)}
        base = {"seed": None, "split": "test", "n_test_subjects": 27, "by_enrolment": {}}
        docs = []
        for i, s in enumerate(SEEDS):
            d = json.loads(json.dumps(base)); d["seed"] = s
            for e, (a, b) in vals.items():
                d["by_enrolment"][e] = {"subject_ids": [str(k) for k in range(27)],
                                        "metrics": {"sequence_top_1_accuracy_1_mins": a + 0.01 * (i - 1),
                                                    "sequence_top_1_accuracy_5_mins": b + 0.01 * (i - 1)}}
            docs.append(d)
        return docs
    ok = read(fake({"all": (0.9, 0.99), "10.0": (0.7, 0.89), "1.0": (0.25, 0.5)}))
    assert ok["verdicts"]["overall"].startswith("PASS"), ok["verdicts"]
    bad = read(fake({"all": (0.9, 0.99), "10.0": (0.7, 0.89), "1.0": (0.50, 0.6)}))
    assert bad["verdicts"]["overall"].startswith("FAIL") and bad["verdicts"]["primary_1min_1min"] == "OUTSIDE", bad["verdicts"]
    flip = read(fake({"all": (0.9, 0.89), "10.0": (0.7, 0.99), "1.0": (0.25, 0.5)}))
    assert flip["verdicts"]["ordering"] == "FAIL" and flip["verdicts"]["overall"].startswith("FAIL")
    print("SELFTEST PASSES: a matching curve passes; a shifted primary cell and a broken ordering both fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(A / "rack2023_read.json"))
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    docs = [json.loads((A / f"rack2023_test_seed{s}.json").read_text()) for s in SEEDS]
    out = read(docs)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=float))      # artefact first
    for n, c in out["cells"].items():
        print(f"  {n:22s} seeds {', '.join(f'{x:.3f}' for x in c['per_seed'])}  mean {c['mean']:.3f} "
              f"range {c['range']:.3f} | published {c['published']:.2f} | inside mean+-range: {c['inside_mean_pm_range']} "
              f"(min-max {c['inside_min_max']}, mean+-sd {c['inside_mean_pm_sd']})")
    g = out["dynamic_range"]
    print(f"  dynamic range all/5 - 1/1: seeds {', '.join(f'{x:.3f}' for x in g['per_seed'])} mean {g['mean']:.3f} "
          f"range {g['range']:.3f} | published 0.73-0.75 | inside: {g['inside_mean_pm_range']}")
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

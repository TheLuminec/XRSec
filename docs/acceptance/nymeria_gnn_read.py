"""
Read the paper_gnn_bilstm arm against nymeria_gnn_REGISTERED.md. Committed before any GNN number exists.

Both arms are the Nymeria in-domain treatment composition (seeds 1-3, the same 3,072 training identities and
48 held-out users); only the extractor differs. Every figure is paired by seed on the same device (cuda).

  1. constrained AUC (script-pair harness, cross-script positives / same-script negatives): per-seed
     difference gnn - bilstm, mean over seeds with a t interval (df = n_seeds - 1). Registered.
  2. rank-1, constrained, N=17, cell-balanced (nymeria_rank1 harness): per (user, script) cell, each arm
     averaged over seeds, difference bootstrapped over users with nymeria_rank1.boot_cells. Registered.
  Reported beside them, not registered: row AUC (the run's own selected_test_auc), each arm's levels, and
  convergence (best_epoch, epochs_run).

    python docs/acceptance/nymeria_gnn_read.py --script-pair docs/acceptance/nymeria_gnn_script_pair.json \
        --rank1 docs/acceptance/nymeria_rank1_{treatment,gnn}_s{1,2,3}_cuda.json [--shard <miami shard>] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105}   # t(0.975, df); no scipy in .venv

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "model"))
from nymeria_rank1 import DRAW_SEED, _cells, boot_cells  # noqa: E402

REGIONS = {
    "constrained_auc_gnn-bilstm": [
        (-np.inf, -0.02, "GNN WORSE: the published graph branches cost the motion signature on AR glasses"),
        (-0.02, 0.02, "BAND: architecture is worth ~0 here too, as on the pooled corpus"),
        (0.02, np.inf, "GNN BETTER: the graph branches add to the motion signature on AR glasses")],
    "rank1_n17_gnn-bilstm": [
        (-np.inf, -0.03, "GNN WORSE on identification"),
        (-0.03, 0.03, "BAND: no identification difference"),
        (0.03, np.inf, "GNN BETTER on identification")],
}
for _n, _rs in REGIONS.items():
    assert _rs[0][0] == -np.inf and _rs[-1][1] == np.inf and all(a[1] == b[0] for a, b in zip(_rs, _rs[1:])), _n
BILSTM, GNN = "treatment", "gnn"


def verdict(name, r):
    hit = [m for a, b, m in REGIONS[name] if r["lo"] < b and r["hi"] >= a]
    at = [m for a, b, m in REGIONS[name] if r["mean"] < b and r["mean"] >= a]
    return f"interval inside one region: {hit[0]}" if len(hit) == 1 else f"interval spans {len(hit)} regions; the mean sits in: {at[0]}"


def t_interval(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return {"mean": float(d.mean()), "lo": float("nan"), "hi": float("nan"), "n": len(d)}
    h = T975[len(d) - 1] * d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "lo": float(d.mean() - h), "hi": float(d.mean() + h), "n": len(d), "per_seed": d.tolist()}


def script_pair(path):
    d = json.loads(pathlib.Path(path).read_text())
    by = {}
    for k, v in d.items():
        arm = BILSTM if v["arm"] == "treatment" else GNN if v["arm"] == "treatment_paper_gnn_bilstm" else None
        if arm is None:
            continue
        assert v["gate_passed"], (k, v["gate_gap"])
        assert v.get("device") == "cuda", (k, v.get("device"))
        assert (arm, int(v["seed"])) not in by, (arm, v["seed"])
        by[(arm, int(v["seed"]))] = v
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-pair", required=True)
    ap.add_argument("--rank1", nargs="+", required=True)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=str(HERE / "nymeria_gnn_read.json"))
    a = ap.parse_args()
    out = {"rows": {}, "verdicts": {}, "levels": {}, "descriptive": {}}
    sp = script_pair(a.script_pair)
    seeds = sorted({s for arm, s in sp if arm == GNN} & {s for arm, s in sp if arm == BILSTM})
    assert seeds, "no seed scored in both arms"
    dc = [sp[(GNN, s)]["constrained_auc"] - sp[(BILSTM, s)]["constrained_auc"] for s in seeds]
    dr = [sp[(GNN, s)]["recorded"] - sp[(BILSTM, s)]["recorded"] for s in seeds]
    for s in seeds:
        assert sp[(GNN, s)]["constrained_pairs"] == sp[(BILSTM, s)]["constrained_pairs"], f"seed {s}: pairs differ"
    out["rows"]["constrained_auc_gnn-bilstm"] = t_interval(dc)
    out["descriptive"]["row_auc_gnn-bilstm"] = t_interval(dr)
    for arm in (BILSTM, GNN):
        out["levels"][arm] = {"constrained_auc": [sp[(arm, s)]["constrained_auc"] for s in seeds],
                              "row_auc": [sp[(arm, s)]["recorded"] for s in seeds]}
    runs = [json.loads(pathlib.Path(p).read_text()) for p in a.rank1]
    refused = [(r["arm"], r["seed"]) for r in runs if "refused" in r]
    by = {(r["arm"], int(r["seed"])): r for r in runs if "refused" not in r}
    rs = sorted({s for arm, s in by if arm == GNN} & {s for arm, s in by if arm == BILSTM})
    assert rs, "no seed rank-1 scored in both arms"
    cells = {arm: [_cells(by[(arm, s)], "constrained", "n17") for s in rs] for arm in (BILSTM, GNN)}
    keys = {u: sorted(v) for u, v in cells[BILSTM][0].items()}
    for arm in cells:
        assert all({u: sorted(v) for u, v in c.items()} == keys for c in cells[arm]), "cells differ"
    users = sorted(keys)
    vec = {arm: [np.array([np.mean([c[u][g] for c in cells[arm]]) for g in keys[u]]) for u in users] for arm in cells}
    rng = np.random.default_rng(DRAW_SEED)
    out["rows"]["rank1_n17_gnn-bilstm"] = {**boot_cells([vec[GNN][i] - vec[BILSTM][i] for i in range(len(users))], rng),
                                          "seeds": rs, "users": len(users), "cells": int(sum(len(v) for v in keys.values()))}
    for arm in (BILSTM, GNN):
        out["levels"][arm]["rank1_n17"] = boot_cells(vec[arm], rng)
    out["refused_rank1"] = refused
    for k, r in out["rows"].items():
        out["verdicts"][k] = verdict(k, r)
    if a.shard:
        rows = [json.loads(l) for l in pathlib.Path(a.shard).read_text().splitlines() if l.strip()]
        conv = {}
        for (arm, s), v in sp.items():
            tail = "/".join(v["checkpoint"].split("/")[:2])
            hit = [r for r in rows if r.get("mode") == "train" and str(r.get("run_dir", "")).rstrip("/").endswith(tail)]
            if len(hit) == 1:
                conv[f"{arm}_s{s}"] = [hit[0].get("best_epoch"), hit[0].get("epochs_run")]
        out["descriptive"]["convergence_best_epoch_epochs_run"] = conv
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=float))      # artefact first
    for arm, l in out["levels"].items():
        print(f"  {arm:9s} constrained AUC {', '.join(f'{x:.4f}' for x in l['constrained_auc'])} | row AUC "
              f"{', '.join(f'{x:.4f}' for x in l['row_auc'])} | rank-1 N=17 {l['rank1_n17']['mean']:.3f} "
              f"[{l['rank1_n17']['lo']:.3f}, {l['rank1_n17']['hi']:.3f}]")
    for k, r in out["rows"].items():
        print(f"  {k:28s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] -> {out['verdicts'][k]}")
    for k, r in out["descriptive"].items():
        print(f"  (descriptive) {k}: {r}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

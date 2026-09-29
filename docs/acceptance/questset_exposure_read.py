"""
Read the reverse-direction arm (questset_exposure_REGISTERED.md) from committed files. Regions,
verdict and bootstrap come from exposure_breadth_read.py (one implementation).

Row 1: Q2 - treatment, Across-XR A1 over all 20 ordered cross-application cells, users 32-48, N=17,
       seed-paired, per-user differences averaged over seeds.
Row 2: Q2 - treatment, Questset GROUP 1 at N=30, per user, seed-paired.
Row 3: Q2 - treatment, each run's own selected_test_auc on the 48 Nymeria users, per seed (±0.02).
Convergence: categorical, best_epoch / epochs_run per seed.

    python docs/acceptance/questset_exposure_read.py --shard <Miami's shard jsonl> [--out ...]
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
from across_xr_alignment_p3 import GAMES, mean_over  # noqa: E402

A = HERE
SEEDS = (1, 2, 3)
ALL_CELLS = [f"{a}->{b}" for a in GAMES for b in GAMES if a != b]
R.ROWS["q2-treatment_axr"] = [(-np.inf, 0.0, "FALSIFIER: Questset exposure does not carry; the Across-XR result is corpus-specific"),
                              (0.0, 0.02, "not resolved"),
                              (0.02, 0.08, "BAND: exposure carries across corpora in both directions"),
                              (0.08, np.inf, "exceeds: report as such")]
R.ROWS["q2-treatment_questset_g1"] = [(-np.inf, 0.0, "FALSIFIER"), (0.0, 0.02, "not resolved"),
                                      (0.02, 0.10, "BAND"), (0.10, np.inf, "exceeds: report as such")]
TREAT_RUN = {1: "09-16-23_train", 2: "12-08-08_train", 3: "14-57-30_train"}
Q2_RUN = {1: "06-16-38_train", 2: "07-49-01_train", 3: "09-21-11_train"}


def treat_axr(s):
    return A / ("exposure_breadth_axr_treatment_s1.json" if s == 1 else f"exposure_breadth_axr_treatment_s{s}_cpu.json")


def gated(p):
    x = load([str(p)])[0]
    assert x["gate"]["passed"], (p, x["gate"])
    return x


def g1_per_user(path):
    d = json.loads(pathlib.Path(path).read_text())
    assert d["gate"]["gap"] < 1e-4, (path, d["gate"]["gap"])
    return d["groups"]["1"]["cells"]["30"]["per_user"]


def row_for(shard_rows, run_dir_tail):
    hits = [r for r in shard_rows if r.get("mode") == "train" and str(r.get("run_dir", "")).rstrip("/").endswith(run_dir_tail)]
    assert len(hits) == 1, (run_dir_tail, len(hits))
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default=str(A / "questset_exposure_read.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(67)
    shard = [json.loads(l) for l in pathlib.Path(a.shard).read_text().splitlines() if l.strip()]
    out = {"per_seed": {}, "pooled": {}, "verdicts": {}}
    d_axr, d_qs, ctrl, conv = [], [], [], {}
    for s in SEEDS:
        q, t = gated(A / f"questset_exposure_axr_s{s}.json"), gated(treat_axr(s))
        dq = mean_over(q, "A1", ALL_CELLS) - mean_over(t, "A1", ALL_CELLS)
        qg1, tg1 = g1_per_user(A / f"questset_exposure_questset_s{s}.json"), g1_per_user(A / f"exposure_breadth_questset_treatment_s{s}_gpu.json")
        users = sorted(tg1); assert sorted(qg1) == users and len(users) == 30
        dg = np.array([qg1[u] - tg1[u] for u in users])
        qr, tr = row_for(shard, "2026-09-29/" + Q2_RUN[s]), row_for(shard, "2026-09-21/" + TREAT_RUN[s])
        assert qr["seed"] == s and tr["seed"] == s and int(qr["num_train_identities"]) == 3072
        c = float(qr["selected_test_auc"]) - float(tr["selected_test_auc"])
        d_axr.append(dq); d_qs.append(dg); ctrl.append(c)
        conv[s] = {"q2": [qr["best_epoch"], qr["epochs_run"]], "treatment": [tr["best_epoch"], tr["epochs_run"]]}
        out["per_seed"][s] = {"axr_A1_q2": float(mean_over(q, "A1", ALL_CELLS).mean()), "axr_A1_treatment": float(mean_over(t, "A1", ALL_CELLS).mean()),
                              "axr_diff": float(dq.mean()), "questset_g1_diff": float(dg.mean()), "control_selected_auc_diff": c,
                              "treatment_axr_device": t["gate"]["device"], "convergence": conv[s]}
        print(f"  seed {s}: Across-XR A1 Q2 {out['per_seed'][s]['axr_A1_q2']:.3f} vs treatment {out['per_seed'][s]['axr_A1_treatment']:.3f} "
              f"({dq.mean():+.3f}) | Questset g1 {dg.mean():+.3f} | own-users AUC {c:+.4f} | best_epoch Q2 {conv[s]['q2']} treatment {conv[s]['treatment']}")
    for k, arr in (("q2-treatment_axr", d_axr), ("q2-treatment_questset_g1", d_qs)):
        r = R.ci(np.mean(arr, axis=0), rng)
        out["pooled"][k] = r
        out["verdicts"][k] = R.verdict(k, r)
    n_out = sum(abs(c) > 0.02 for c in ctrl)
    out["pooled"]["control_own_users"] = {"per_seed": ctrl, "outside_0.02": int(n_out)}
    out["verdicts"]["control_own_users"] = ("holds: within ±0.02 in every seed" if n_out == 0 else
                                            "noted: outside in exactly 1 of 3" if n_out == 1 else
                                            "the swap moved in-domain in 2+ seeds: read row 1 with that")
    stop_q = [s for s in SEEDS if conv[s]["q2"][1] < 120]; stop_t = [s for s in SEEDS if conv[s]["treatment"][1] < 120]
    out["verdicts"]["convergence"] = f"patience stops: Q2 seeds {stop_q}, treatment seeds {stop_t}"
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print()
    for k in ("q2-treatment_axr", "q2-treatment_questset_g1"):
        r = out["pooled"][k]
        print(f"  {k:26s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] -> {out['verdicts'][k]}")
    print(f"  control own users {['%+.4f' % c for c in ctrl]} -> {out['verdicts']['control_own_users']}")
    print(f"  {out['verdicts']['convergence']}\nwrote {a.out}")


if __name__ == "__main__":
    main()

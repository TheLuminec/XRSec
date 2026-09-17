"""
Review addenda to the Schach paired comparison (paper review, 2026-09-17).

Nothing here re-embeds or re-scores anything. It reads the per-user arrays already in the
committed certificates and computes four things the paper draft quoted without a certificate,
or quoted from a certificate whose definition differs from what Section 5.3 of the draft says:

  1. The two cross-arm contrasts that `across_xr_alignment_aggregate.py` pairs BY SEED ORDER,
     truncating to the shorter arm - so Z-676 (one seed) minus zero-shot and C2-hi (one seed)
     minus C2-lo are paired to SEED 1 of the three-seed arm. Recomputed against the three-seed
     arm's per-user seed MEAN, which is the unit Section 5.3 of the draft describes.
  2. The ten-minute `raw` versus `dyn` differences (our majority vote), paired per user, for
     both the zero-shot arm (3 v 3 seeds) and the exposed arm (3 dyn seeds v 1 raw seed, and
     seed 1 v seed 1). The draft quoted the exposed reversal as two levels with no interval.
  3. Per-user standard deviations with ddof=1 (the certificate's `sd_users` convention and
     Amendment 8's), under their metric, for the three systems in Section 6.9.
  4. The template-averaging gains (our-metric level minus their-metric level) at full precision.

Gate, asserted before anything is written: the script must reproduce the aggregate
certificate's own seed-1 cross contrasts and the paired certificate's headline contrast and
`sd_users` from the same arrays. If it cannot, the arrays are not the ones the certificates
were built from and nothing below is quotable.

    python docs/acceptance/schach_paired_review_addenda.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import across_xr_alignment as axa  # noqa: E402

N_BOOT = 10000


def a1(name: str) -> np.ndarray:
    d = json.loads((HERE / f"across_xr_alignment_{name}.json").read_text(encoding="utf-8"))
    s = d["seeds"][0]
    assert s.get("gate", {}).get("passed") is True, f"{name}: gate not passed"
    v = np.asarray(s["arms"]["A1"]["per_user"], dtype=float)
    assert v.shape == (17,), (name, v.shape)
    return v


def iv(values: np.ndarray, rng: np.random.Generator) -> dict:
    mean, lo, hi = axa.bootstrap_mean(np.asarray(values, dtype=float), N_BOOT, rng)
    return {"mean": mean, "ci95_boot": [lo, hi]}


def main() -> int:
    rng = np.random.default_rng(67)
    paired = json.loads((HERE / "schach_paired.json").read_text(encoding="utf-8"))
    agg = json.loads((HERE / "across_xr_alignment_aggregate.json").read_text(encoding="utf-8"))

    zs = [a1(f"seed{s}") for s in (1, 2, 3)]
    c2 = [a1(f"c2lo_seed{s}") for s in (1, 2, 3)]
    z676, c2hi = a1("z676_seed1"), a1("c2hi_seed1")
    zs_mean, c2_mean = np.mean(zs, axis=0), np.mean(c2, axis=0)

    # ---- gate: the arrays reproduce the certificates they came from -----------------------
    gate = {}
    gate["aggregate_Z676-A1_seed1_mean"] = {"certificate": agg["cross"]["Z676-A1"]["mean"],
                                            "recomputed": float((z676 - zs[0]).mean())}
    gate["aggregate_C2hi-C2lo_seed1_mean"] = {"certificate": agg["cross"]["C2hi-C2lo"]["mean"],
                                              "recomputed": float((c2hi - c2[0]).mean())}
    gate["aggregate_C2lo-A1_3seed_mean"] = {"certificate": agg["cross"]["C2lo-A1"]["mean"],
                                            "recomputed": float((c2_mean - zs_mean).mean())}
    th = np.asarray(paired["theirs_D1"]["per_user"], dtype=float)
    c2_d1 = np.asarray(paired["ours"]["c2lo"]["D1"]["per_user"], dtype=float)
    gate["paired_C2lo-theirs_D1_mean"] = {"certificate": paired["ours"]["c2lo"]["vs_theirs_D1"]["paired"]["mean"],
                                          "recomputed": float((c2_d1 - th).mean())}
    gate["paired_theirs_sd_users_ddof1"] = {"certificate": paired["theirs_D1"]["level"]["sd_users"],
                                            "recomputed": float(th.std(ddof=1))}
    for k, v in gate.items():
        assert abs(v["certificate"] - v["recomputed"]) < 1e-9, (k, v)
        v["passed"] = True
    # The certificate's own A1 per-seed means must be the arrays' means (same arrays, not copies).
    assert paired["ours"]["c2lo"]["seeds_scored"] == [1, 2, 3] and paired["ours"]["zero_shot"]["seeds_scored"] == [1, 2, 3]
    for arm, vecs in (("c2lo", c2), ("zero_shot", zs)):
        for s, v in zip((1, 2, 3), vecs):
            cpu_rescore = paired["ours"][arm]["D2"]["per_seed_mean"][s - 1]
            assert abs(cpu_rescore - v.mean()) < 2e-3, (arm, s, cpu_rescore, v.mean())   # CPU/GPU tolerance
    gate["alignment_vs_paired_per_seed_A1_within_2e-3"] = {"passed": True, "arms": ["zero_shot", "c2lo"]}

    out = {"what": "review addenda computed from committed certificates only; see module docstring",
           "gate": gate, "n_boot": N_BOOT, "rng_seed": 67}

    # ---- 1. cross-arm contrasts against the three-seed per-user mean -----------------------
    out["cross_vs_three_seed_mean"] = {
        "Z676-zero_shot": {**iv(z676 - zs_mean, rng), "note": "aggregate.json pairs to seed 1 only: -0.013 [-0.039, +0.012]"},
        "C2hi-C2lo": {**iv(c2hi - c2_mean, rng), "note": "aggregate.json pairs to seed 1 only: -0.061 [-0.099, -0.026]"},
        "C2lo-zero_shot": iv(c2_mean - zs_mean, rng),
        "C2hi-Z676": iv(c2hi - z676, rng),
    }

    # ---- 2. ten-minute raw v dyn, our vote, paired per user ---------------------------------
    def seq_user(arm: str, seed: int) -> np.ndarray:
        return np.asarray(paired["ours"][arm][f"seed{seed}"]["D2_seq10_per_cell_user"], dtype=float).mean(axis=0)

    def p1_user(arm: str, seed: int) -> np.ndarray:
        return np.asarray(paired["ours"][arm][f"seed{seed}"]["D2_p1_per_cell_user"], dtype=float).mean(axis=0)

    zs_seq = np.mean([seq_user("zero_shot", s) for s in (1, 2, 3)], axis=0)
    raw_seq = np.mean([seq_user("raw", s) for s in (1, 2, 3)], axis=0)
    c2_seq = np.mean([seq_user("c2lo", s) for s in (1, 2, 3)], axis=0)
    c2raw_seq = seq_user("c2lo_raw", 1)
    out["ten_minute_raw_minus_dyn_our_vote"] = {
        "zero_shot_3v3": {**iv(raw_seq - zs_seq, rng), "raw_level": float(raw_seq.mean()), "dyn_level": float(zs_seq.mean())},
        "exposed_1v3": {**iv(c2raw_seq - c2_seq, rng), "raw_level": float(c2raw_seq.mean()), "dyn_level": float(c2_seq.mean())},
        "exposed_1v1_seed1": {**iv(c2raw_seq - seq_user("c2lo", 1), rng), "dyn_seed1_level": float(seq_user("c2lo", 1).mean())},
    }
    # single-window check against p2.json's definition (exposed raw seed 1 v dyn three seeds)
    c2_p1 = np.mean([p1_user("c2lo", s) for s in (1, 2, 3)], axis=0)
    out["single_window_exposed_raw_minus_dyn_check"] = iv(p1_user("c2lo_raw", 1) - c2_p1, rng)

    # ---- 3. per-user sd, ddof=1, their metric ---------------------------------------------
    out["per_user_sd_ddof1_their_metric"] = {
        "theirs": float(th.std(ddof=1)),
        "zero_shot": float(np.asarray(paired["ours"]["zero_shot"]["D1"]["per_user"]).std(ddof=1)),
        "c2lo": float(c2_d1.std(ddof=1)),
        "ddof0_for_reference": {"theirs": float(th.std(ddof=0)),
                                "zero_shot": float(np.asarray(paired["ours"]["zero_shot"]["D1"]["per_user"]).std(ddof=0)),
                                "c2lo": float(c2_d1.std(ddof=0))},
    }

    # ---- 4. template-averaging gains at full precision ------------------------------------
    out["template_gain_our_metric_minus_their_metric"] = {
        "theirs": paired["theirs_D2"]["level"]["mean"] - paired["theirs_D1"]["level"]["mean"],
        "zero_shot": paired["ours"]["zero_shot"]["D2"]["level"]["mean"] - paired["ours"]["zero_shot"]["D1"]["level"]["mean"],
        "c2lo": paired["ours"]["c2lo"]["D2"]["level"]["mean"] - paired["ours"]["c2lo"]["D1"]["level"]["mean"],
    }

    path = HERE / "schach_paired_review_addenda.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

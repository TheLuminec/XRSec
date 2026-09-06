"""
Does window length buy rank-1, at MATCHED total evidence?

k and window length both buy seconds, so 10 s at k=16 against 5 s at k=16 would be 160 s of
enrolment against 80 s and the comparison would conflate the two - the same confound as
scoring ViewGauss at k=3 against BOXRR at k=16, and the one window_stride exists to separate.
The design that isolates window length is 5 s at k=16 against 10 s at k=8, both 80 s.

Two details that decide whether "80 s" is true:

  - The 10 s checkpoints TRAINED at window_stride=5, so their windows overlap by half. Eight
    of those span 45 s of wall clock, not 80. Scoring does not have to inherit the training
    layout - the model consumes one window at a time - so both indices are built at full
    stride and 80 s means 80 distinct seconds on both arms.
  - The population is the INTERSECTION of users passing both gates, so the two arms score
    the same people and the contrast is paired.

Registered by the Coordinator beforehand: under +0.05 at matched evidence, since window
length was worth +0.019 AUC on verification and the k-curve says evidence rather than window
structure is what moves rank-1. Above +0.10 would be a real finding.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index                 # noqa: E402
from normalization import ChannelNormalizer            # noqa: E402
from utils import load_checkpoint                      # noqa: E402
from score_nymeria import gate                         # noqa: E402
from step6_seated_dyn import (DYN_93, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, cosine, rank1)
from step6_clean_boxrr import BOXRR, unseen_pool       # noqa: E402
from step6_implied_rank1 import auc_from, implied_rank1  # noqa: E402

TEN_S = {  # seed -> the 419-identity 10 s dyn checkpoint (sweep 0840769514)
    1: "sweeps/0840769514/runs/bilstm_a41190094c/best.pth",
    2: "sweeps/0840769514/runs/bilstm_0ebccac678/best.pth",
    3: "sweeps/0840769514/runs/bilstm_ab82c3b90b/best.pth",
    4: "sweeps/0840769514/runs/bilstm_796d3932d4/best.pth",
    5: "sweeps/0840769514/runs/bilstm_8d679a46cc/best.pth",
}
ARMS = {  # label -> (checkpoints, sample_time, seq_len, k)   both arms are 80 s of evidence
    "5s  k=16": (DYN_93, 5, 100, 16),
    "10s k=8 ": (TEN_S, 10, 200, 8),
}


def arm_population(users, sample_time, k):
    kw = dict(sample_time=sample_time, sample_rate=20, exclude_users=users, swap_data=True)
    raw = quiet(build_sample_index, str(BOXRR.resolve()), **kw)
    return raw, {u: (g, p) for u, g, p in population(raw, k)}


if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N={N_SMALL} is {1/N_SMALL:.4f}\n")

    for ckpts, st, seq, _ in ARMS.values():
        for seed, path in sorted(ckpts.items()):
            g = gate(path)
            if not g["passed"]:
                print("\n*** GATE FAILED ***")
                sys.exit(1)
    print("\ncheckpoint gate: 10/10 passed\n")

    users = unseen_pool()
    pops, raws = {}, {}
    for label, (_, st, _, k) in ARMS.items():
        raws[label], pops[label] = arm_population(users, st, k)
        print(f"  {label}: {len(pops[label])} users pass the k={k} gate at {st}s")
    shared = sorted(set.intersection(*(set(p) for p in pops.values())))
    print(f"  paired population: {len(shared)} users in both\n")

    out = {}
    for label, (ckpts, st, seq, k) in ARMS.items():
        pop = [(u, *pops[label][u]) for u in shared]
        kw = dict(sample_time=st, sample_rate=20, exclude_users=users, swap_data=True)
        per_seed = {"rank1": [], "auc": [], "implied": []}
        for seed, path in sorted(ckpts.items()):
            model, ck = load_checkpoint(path, DEVICE, seq, return_checkpoint=True)
            dyn = quiet(build_sample_index, str(BOXRR.resolve()), encoding="dyn", **kw)
            quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
            emb = embed(model, dyn.samples)
            g_e, p_e, o = templates(emb, pop, k, np.random.default_rng(SEED))
            d = cosine(p_e, g_e)
            mask = np.zeros_like(d, dtype=bool)
            mask[np.arange(len(d)), o] = True
            auc = auc_from(-d[mask], -d[~mask])
            per_seed["rank1"].append(rank1(d, o, N_SMALL, np.random.default_rng(SEED)))
            per_seed["auc"].append(auc)
            per_seed["implied"].append(implied_rank1(auc))
        out[label] = per_seed
        print(f"  {label} (80s)   rank-1 {np.mean(per_seed['rank1']):.3f}+-{np.std(per_seed['rank1']):.3f}"
              f"   AUC {np.mean(per_seed['auc']):.4f}   implied {np.mean(per_seed['implied']):.3f}"
              f"   offset {np.mean(per_seed['rank1']) - np.mean(per_seed['implied']):+.3f}", flush=True)

    a, b = out["5s  k=16"]["rank1"], out["10s k=8 "]["rank1"]
    paired = np.array(b) - np.array(a)          # seeds are matched: same seed, same users
    diff = float(paired.mean())
    t = diff / (paired.std(ddof=1) / np.sqrt(len(paired))) if paired.std(ddof=1) else float("inf")
    print(f"\n  10s - 5s at matched 80s evidence: {diff:+.3f}  (paired sd {paired.std(ddof=1):.3f}, "
          f"t({len(paired)-1})={t:.2f}, won {int((paired > 0).sum())}/{len(paired)})")
    print(f"  registered: under +0.05 expected, above +0.10 a real finding -> "
          + ("REAL FINDING" if diff > 0.10 else "as predicted" if diff < 0.05 else "BETWEEN - neither"))
    pathlib.Path("docs/acceptance/step6_window_length_rank1.json").write_text(
        json.dumps({"users": len(shared), "arms": out, "paired_diff": diff}, indent=1), encoding="utf-8")

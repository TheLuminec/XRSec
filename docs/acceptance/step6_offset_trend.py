"""
Does the rank-1 offset shrink by MORE than the ceiling forces? The well-posed version.

The four-point sequence 0.116 / 0.107 / 0.074 / 0.046 was refused because the offset is
bounded above by the headroom `1 - implied`, which collapses as AUC rises, and because
normalising by that headroom reverses the direction while itself moving with enrolment
evidence. Matching arms on AUC by varying k re-introduces exactly that evidence dependence,
so that route is closed.

What is not closed is holding population, evidence and window length ALL fixed and varying
only identity count. Two parts, per the Coordinator:

  PART 1  Score the 419 and the 2096 checkpoints on the SAME clean users at 10 s, k=8. The
          pool is BOXRR users outside BOTH arms' subsamples, so neither has trained on or
          validated against any of them. AUC level is then the only difference between the
          arms and the question is well-posed rather than confounded.

  PART 2  The decisive post-hoc form. Take the 419 arm's own genuine and impostor scores and
          apply a monotone map to the GENUINE side alone until its AUC equals the 2096 arm's,
          then recompute rank-1. A monotone map applied to every score leaves AUC untouched -
          AUC is rank-based - so the map has to act on one side, and what it preserves is the
          shape of that side. If the rescaled 419 rank-1 lands on the measured 2096 rank-1,
          the shape never changed and the ceiling did all the work. If the measured one is
          higher, the distribution really did become less Gaussian with identity count.

          The map is not unique, so three are used - shift, scale, and a quantile stretch -
          and a conclusion is only taken if all three agree. That is the stated weakness of
          the design and the reason for running more than one.

PREDICTION, registered by committing this file before it is run.

  More training identities should help the HARD users most, since that is where finer
  distinctions are needed, which would REDUCE the per-user heterogeneity that produces the
  offset in the first place. So I expect the measured 2096 rank-1 to be at or BELOW the
  rescaled 419 value: the ceiling explains the shrinkage, and possibly more than all of it.

  Registered outcome: measured_2096 - rescaled_419 <= +0.02 under all three maps.
  Falsifier: measured_2096 exceeds rescaled_419 by more than +0.02 under all three maps,
  which would say the score distribution became LESS Gaussian as identities grew - the
  opposite of my mechanism, and the more interesting result.
  Anything that splits across the three maps is "not resolved", and the maps disagreeing is
  itself the finding that the design cannot answer the question.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.optimize import brentq

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index, select_user_subset   # noqa: E402
from normalization import ChannelNormalizer                  # noqa: E402
from utils import load_checkpoint                            # noqa: E402
from step6_seated_dyn import (DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, cosine, rank1)
from step6_clean_boxrr import BOXRR, POOL_USERS              # noqa: E402
from step6_implied_rank1 import auc_from, implied_rank1      # noqa: E402

ARMS = {
    "419 ids":  {1: "sweeps/0840769514/runs/bilstm_a41190094c/best.pth",
                 2: "sweeps/0840769514/runs/bilstm_0ebccac678/best.pth"},
    "2096 ids": {1: "sweeps/b4617a5f05/runs/bilstm_a41190094c/best.pth",
                 2: "sweeps/ff5a5dd1b0/runs/bilstm_0ebccac678/best.pth"},
}
K, SEQ, ST = 8, 200, 10


def subsample_union(ckpts: dict) -> set[str]:
    used: set[str] = set()
    for seed, path in sorted(ckpts.items()):
        es = torch.load(path, map_location="cpu", weights_only=False)["eval_split"]
        keep = select_user_subset(es["data_dirs"], es.get("max_users"), seed)
        if keep is None:
            raise SystemExit(f"{path} trained on the whole corpus; no clean pool")
        used |= {str(pathlib.Path(u).resolve()) for u in keep}
    return used


def score_matrix(path, users, pop):
    model, ck = load_checkpoint(path, DEVICE, SEQ, return_checkpoint=True)
    dyn = quiet(build_sample_index, str(BOXRR.resolve()), sample_time=ST, sample_rate=20,
                exclude_users=users, swap_data=True, encoding="dyn")
    quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
    emb = embed(model, dyn.samples)
    g_e, p_e, o = templates(emb, pop, K, np.random.default_rng(SEED))
    return cosine(p_e, g_e), o


def split(d, owners):
    mask = np.zeros_like(d, dtype=bool)
    mask[np.arange(len(d)), owners] = True
    return mask


#: Monotone maps on the GENUINE side. Each takes the genuine scores (cosine DISTANCES, so
#: smaller is more similar) and a parameter, and must be increasing in the parameter's effect
#: on AUC. They differ in what they preserve: a shift keeps every gap, a scale keeps every
#: ratio, a quantile stretch keeps only the rank order.
MAPS = {
    "shift": lambda g, t: g - t,
    "scale": lambda g, t: g * np.exp(-t),
    "stretch": lambda g, t: np.mean(g) + (g - np.mean(g)) * np.exp(-t) - t,
}


def rescale_to(d, owners, target_auc, fn):
    """Apply `fn` to the genuine cells until the whole matrix's AUC equals target_auc."""
    mask = split(d, owners)
    impostor = -d[~mask]
    base = d[mask].copy()

    def gap(t):
        return auc_from(-fn(base, t), impostor) - target_auc

    lo, hi = 0.0, 1e-3
    while gap(hi) < 0 and hi < 1e3:
        hi *= 2.0
    t = brentq(gap, lo, hi, xtol=1e-12) if gap(lo) <= 0 <= gap(hi) else 0.0
    out = d.copy()
    out[mask] = fn(base, t)
    return out, auc_from(-out[mask], -out[~mask])


if __name__ == "__main__":
    pool_used = set().union(*(subsample_union(c) for c in ARMS.values()))
    every = sorted(str(p.resolve()) for p in BOXRR.iterdir() if p.is_dir())
    clean = [u for u in every if u not in pool_used]
    print(f"BOXRR {len(every)} users; in either arm's subsample {len(pool_used)}; "
          f"clean for BOTH {len(clean)}")
    users = list(np.random.default_rng(SEED).choice(clean, min(POOL_USERS, len(clean)), replace=False))

    raw = quiet(build_sample_index, str(BOXRR.resolve()), sample_time=ST, sample_rate=20,
                exclude_users=users, swap_data=True)
    pop = population(raw, K)
    print(f"population {len(pop)} users, {ST}s k={K} (80 s), chance {1/N_SMALL:.4f}\n")

    got, mats = {}, {}
    for label, ckpts in ARMS.items():
        rows = {"auc": [], "rank1": [], "implied": [], "offset": []}
        for seed, path in sorted(ckpts.items()):
            d, o = score_matrix(path, users, pop)
            mask = split(d, o)
            auc = auc_from(-d[mask], -d[~mask])
            r1, imp = rank1(d, o, N_SMALL, np.random.default_rng(SEED)), implied_rank1(auc)
            for k, v in (("auc", auc), ("rank1", r1), ("implied", imp), ("offset", r1 - imp)):
                rows[k].append(v)
            mats.setdefault(label, []).append((d, o))
        got[label] = rows
        print(f"  {label:<9} AUC {np.mean(rows['auc']):.4f}   implied {np.mean(rows['implied']):.3f}"
              f"   measured {np.mean(rows['rank1']):.3f}   offset {np.mean(rows['offset']):+.3f}",
              flush=True)

    target = float(np.mean(got["2096 ids"]["auc"]))
    measured = float(np.mean(got["2096 ids"]["rank1"]))
    print(f"\npart 2: rescaling the 419 scores to the 2096 AUC of {target:.4f}")
    print(f"        measured 2096 rank-1 is {measured:.3f}\n")
    verdicts = {}
    for name, fn in MAPS.items():
        vals = []
        for d, o in mats["419 ids"]:
            resc, auc_check = rescale_to(d, o, target, fn)
            assert abs(auc_check - target) < 1e-6, f"{name} failed to hit the target AUC"
            vals.append(rank1(resc, o, N_SMALL, np.random.default_rng(SEED)))
        r = float(np.mean(vals))
        verdicts[name] = measured - r
        print(f"  {name:<9} rescaled 419 rank-1 {r:.3f}   measured - rescaled {measured - r:+.3f}")

    deltas = np.array(list(verdicts.values()))
    if (deltas <= 0.02).all():
        verdict = "AS PREDICTED - the ceiling accounts for the shrinkage; shape unchanged"
    elif (deltas > 0.02).all():
        verdict = "FALSIFIED - the distribution became LESS Gaussian as identities grew"
    else:
        verdict = "NOT RESOLVED - the three maps disagree, so this design cannot answer it"
    print(f"\n{verdict}")
    pathlib.Path("docs/acceptance/step6_offset_trend.json").write_text(json.dumps(
        {"users": len(pop), "arms": got, "target_auc": target, "measured_2096": measured,
         "rescaled_deltas": verdicts, "verdict": verdict}, indent=1), encoding="utf-8")

"""
Does the rank-1 offset repeat at a second and third identity count?

The offset - measured rank-1 minus what the same score set's verification AUC implies under
an equal-variance Gaussian - was +0.109 at 419 identities and 5 s, and +0.108 at 10 s. It is a
SHAPE property of one score set: implied and measured come out of the same distance matrix on
the same users, so it is a within-run quantity and does not depend on the level being clean.

That distinction decides which population each arm can use:

  2096 identities  trained at max_users=2020, so 2000 of the 4020 BOXRR users are in neither
                   draw. Clean pool, no caveat, level and offset both reportable.
  4096 identities  trained at max_users=None, so EVERY BOXRR user is inside the subsample and
                   no clean pool exists. Its validation users are the only population, and
                   they chose the epoch - so the OFFSET is the primary quantity here and the
                   absolute rank-1 is labelled a validation-user figure wherever it appears.
                   Selection can lift AUC and rank-1 together; it does not obviously distort
                   the gap between them.

The 4096 arm's implied value is 0.785, numerically the published figure this project keeps
being compared against. Landing there is NOT news - it was implied by a verification number
already in hand. Registered beforehand: +0.11 predicts about 0.89.
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

from dataset import build_sample_index, select_user_subset   # noqa: E402
from normalization import ChannelNormalizer                  # noqa: E402
from utils import load_checkpoint                            # noqa: E402
from score_nymeria import gate                               # noqa: E402
from step6_seated_dyn import (DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, cosine, rank1)
from step6_clean_boxrr import BOXRR, POOL_USERS              # noqa: E402
from step6_implied_rank1 import auc_from, implied_rank1      # noqa: E402
from step6_indomain_dyn import validation_users              # noqa: E402

ARMS = {
    "2096 ids, 10s": {1: "sweeps/b4617a5f05/runs/bilstm_a41190094c/best.pth",
                      2: "sweeps/ff5a5dd1b0/runs/bilstm_0ebccac678/best.pth"},
    "4096 ids, 10s": {1: "sweeps/f9ca1571b9/runs/bilstm_a41190094c/best.pth",
                      2: "sweeps/c05d670fff/runs/bilstm_0ebccac678/best.pth"},
}
K, SEQ = 8, 200          # 8 windows of 10 s = 80 s, matching every other column here


def pool_for(ckpts: dict) -> tuple[list[str], str]:
    """BOXRR users outside every checkpoint's subsample. Falls back to the validation draw,
    and SAYS SO, when the arm trained on the whole corpus and no clean pool exists."""
    used: set[str] = set()
    for seed, path in sorted(ckpts.items()):
        es = torch.load(path, map_location="cpu", weights_only=False)["eval_split"]
        keep = select_user_subset(es["data_dirs"], es.get("max_users"), seed)
        if keep is None:
            users = sorted(set().union(*(
                set(validation_users(torch.load(p, map_location="cpu",
                                                weights_only=False)["eval_split"], s, BOXRR))
                for s, p in sorted(ckpts.items()))))
            return users, "VALIDATION USERS (they chose the epoch; offset is the primary quantity)"
        used |= {str(pathlib.Path(u).resolve()) for u in keep}
    every = sorted(str(p.resolve()) for p in BOXRR.iterdir() if p.is_dir())
    pool = [u for u in every if u not in used]
    chosen = list(np.random.default_rng(SEED).choice(pool, min(POOL_USERS, len(pool)), replace=False))
    return chosen, f"CLEAN ({len(pool)} of {len(every)} users in neither draw)"


if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N={N_SMALL} is {1/N_SMALL:.4f}; k={K} (80 s)\n")
    out = {}
    for label, ckpts in ARMS.items():
        for seed, path in sorted(ckpts.items()):
            if not gate(path)["passed"]:
                print(f"\n*** GATE FAILED on {path} ***")
                sys.exit(1)
        users, provenance = pool_for(ckpts)
        kw = dict(sample_time=10, sample_rate=20, exclude_users=users, swap_data=True)
        raw = quiet(build_sample_index, str(BOXRR.resolve()), **kw)
        pop = population(raw, K)
        print(f"\n{label}: population {len(pop)} users - {provenance}")

        per_seed = {"rank1": [], "auc": [], "implied": [], "offset": []}
        for seed, path in sorted(ckpts.items()):
            model, ck = load_checkpoint(path, DEVICE, SEQ, return_checkpoint=True)
            dyn = quiet(build_sample_index, str(BOXRR.resolve()), encoding="dyn", **kw)
            quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
            emb = embed(model, dyn.samples)
            g_e, p_e, o = templates(emb, pop, K, np.random.default_rng(SEED))
            d = cosine(p_e, g_e)
            mask = np.zeros_like(d, dtype=bool)
            mask[np.arange(len(d)), o] = True
            auc = auc_from(-d[mask], -d[~mask])
            r1, imp = rank1(d, o, N_SMALL, np.random.default_rng(SEED)), implied_rank1(auc)
            for key, v in (("rank1", r1), ("auc", auc), ("implied", imp), ("offset", r1 - imp)):
                per_seed[key].append(v)
            print(f"    seed {seed}   AUC {auc:.4f}   implied {imp:.3f}   measured {r1:.3f}"
                  f"   offset {r1 - imp:+.3f}", flush=True)
        out[label] = {"provenance": provenance, "users": len(pop), **per_seed}
        print(f"    mean      AUC {np.mean(per_seed['auc']):.4f}   "
              f"implied {np.mean(per_seed['implied']):.3f}   "
              f"measured {np.mean(per_seed['rank1']):.3f}   "
              f"OFFSET {np.mean(per_seed['offset']):+.3f}")
    pathlib.Path("docs/acceptance/step6_identity_count_offset.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")

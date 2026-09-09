"""
The clean version of the 0.858: BOXRR users that were in NEITHER draw.

The headline in-domain figure was measured on each checkpoint's validation users, and those
users chose the epoch, which CLAUDE.md prices at about +0.02. That qualification is exactly
the kind that gets dropped the third time a number is quoted, so this removes the need for
it rather than restating it.

The pool is BOXRR users outside the union of all five checkpoints' subsamples - never in
any training set, never in any validation draw, never having influenced any epoch choice.
It is also the SAME pool for all five checkpoints, so the seed spread here is the model and
nothing else; the validation-user figure could not say that, because each seed scored its
own different users.

Population size is deliberately held near the validation figure's 73-92 users. A much
larger pool would change the impostor diversity of the N=17 draws and make the two columns
answer slightly different questions.

Gated like everything else: each checkpoint reproduces its own recorded metric on its own
recorded users before it is allowed to produce a rank-1.
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

from dataset import build_sample_index, select_user_subset       # noqa: E402
from normalization import ChannelNormalizer                      # noqa: E402
from utils import load_checkpoint                                # noqa: E402
from score_nymeria import gate                                   # noqa: E402
from step6_seated_dyn import (write_gate_certificate, DYN_93, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, euclid, cosine, rank1, zscore)

BOXRR = ROOT / "processed_datasets" / "BOXRR-23_Dataset" / "users"
KS = (1, 4, 16)
K_POP = 16
POOL_USERS = 100          # gated down to ~the validation figure's 73-92


def unseen_pool() -> list[str]:
    """BOXRR users outside every checkpoint's subsample - so outside every training set AND
    every validation draw, for all five seeds at once."""
    used: set[str] = set()
    for seed, path in sorted(DYN_93.items()):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        es = ck["eval_split"]
        keep = select_user_subset(es["data_dirs"], es.get("max_users"), seed)
        if keep is None:
            raise SystemExit("a checkpoint used the whole corpus; there is no unseen pool")
        used |= {str(pathlib.Path(u).resolve()) for u in keep}
    every = sorted(str(p.resolve()) for p in BOXRR.iterdir() if p.is_dir())
    pool = [u for u in every if u not in used]
    print(f"BOXRR users {len(every)}; in some subsample {len(used)}; never used {len(pool)}")
    return list(np.random.default_rng(SEED).choice(pool, min(POOL_USERS, len(pool)), replace=False))


if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N={N_SMALL} is {1/N_SMALL:.4f}\n")

    gates = [dict(gate(p), seed=s) for s, p in sorted(DYN_93.items())]
    write_gate_certificate(pathlib.Path(__file__).stem, gates)
    if not all(g["passed"] for g in gates):
        print("\n*** GATE FAILED - report the mismatch, do not compute rank-1 ***")
        sys.exit(1)
    print("\ngate: 5/5 passed\n")

    users = unseen_pool()
    kw = dict(sample_time=5, sample_rate=20, exclude_users=users, swap_data=True)
    raw = quiet(build_sample_index, str(BOXRR.resolve()), **kw)
    pop = population(raw, K_POP)
    pos = raw.samples[:, 4:7, :].mean(dim=2).numpy()
    print(f"population {len(pop)} users at k={K_POP} (validation-user column had 73-92)\n")

    per_k = {k: {"dyn": [], "y": [], "y+dyn": []} for k in KS}
    for seed, path in sorted(DYN_93.items()):
        model, ck = load_checkpoint(path, DEVICE, 100, return_checkpoint=True)
        dyn = quiet(build_sample_index, str(BOXRR.resolve()), encoding="dyn", **kw)
        assert torch.equal(dyn.window_session_ids, raw.window_session_ids)
        quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
        emb = embed(model, dyn.samples)
        for k in KS:
            g_e, p_e, o = templates(emb, pop, k, np.random.default_rng(SEED))
            g_p, p_p, _ = templates(pos, pop, k, np.random.default_rng(SEED))
            d_dyn, d_y = cosine(p_e, g_e), euclid(p_p, g_p, [1])
            for lab, d in (("dyn", d_dyn), ("y", d_y), ("y+dyn", zscore(d_dyn) + zscore(d_y))):
                per_k[k][lab].append(rank1(d, o, N_SMALL, np.random.default_rng(SEED)))

    for k in KS:
        print(f"  k={k:<3} ({k*5:>2}s)   "
              + "   ".join(f"{lab} {np.mean(v):.3f}+-{np.std(v):.3f}" for lab, v in per_k[k].items()),
              flush=True)
    out = {"users": len(pop), "pool": len(users), **{str(k): v for k, v in per_k.items()}}
    pathlib.Path("docs/acceptance/step6_clean_boxrr.json").write_text(json.dumps(out, indent=1),
                                                                      encoding="utf-8")

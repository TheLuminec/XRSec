"""
The reference column the seated numbers need: the SAME five checkpoints on the activities
they were trained on, at the same N and the same k.

Without this the seated dyn figures have nothing to be read against. "0.18 at N=17" is
three times chance and could be read either as a weak signal or as a collapse, and only the
in-domain figure from the same checkpoints decides which.

Users are each checkpoint's own held-out validation draw - never anyone it trained on -
recovered with the recipe the step 6 calibration gate already reproduced digit-exactly.
BOXRR is the training activity; alyx is the other training corpus and is the one whose
sessions are days apart.
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

from dataset import select_user_subset, select_validation_users, build_sample_index  # noqa: E402
from normalization import ChannelNormalizer                                          # noqa: E402
from utils import load_checkpoint                                                     # noqa: E402
from step6_seated_dyn import (DYN_93, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, euclid, cosine, rank1, zscore)

CORPORA = {
    "BOXRR held-out":  ROOT / "processed_datasets" / "BOXRR-23_Dataset" / "users",
    "alyx held-out":   ROOT / "processed_datasets" / "who_is_alyx" / "users",
}
K = 16


def validation_users(eval_split: dict, seed: int, corpus: pathlib.Path) -> list[str]:
    """The in-domain validation draw, restricted to the subsample and to one corpus. This
    is NOT eval_split's user list - that is the test group; the validation draw is not
    stored on the checkpoint and has to be recomputed from the recorded settings."""
    dirs = eval_split["data_dirs"]
    keep = select_user_subset(dirs, eval_split.get("max_users"), seed)
    val = select_validation_users(dirs, eval_split.get("exclude_users") or [], 0.25, seed)
    if keep is not None:
        val = [u for u in val if u in set(keep)]
    return [u for u in val if pathlib.Path(u).parent.resolve() == corpus.resolve()]


if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N={N_SMALL} is {1/N_SMALL:.4f}; k={K}\n")
    results = {}
    for name, corpus in CORPORA.items():
        per_seed = {"dyn": [], "y": [], "y+dyn": []}
        sizes = []
        for seed, path in sorted(DYN_93.items()):
            model, ck = load_checkpoint(path, DEVICE, 100, return_checkpoint=True)
            users = validation_users(ck["eval_split"], seed, corpus)
            if not users:
                print(f"  {name}: seed {seed} has no validation users in this corpus")
                continue
            kw = dict(sample_time=5, sample_rate=20, exclude_users=users, swap_data=True)
            dyn = quiet(build_sample_index, str(corpus.resolve()), encoding="dyn", **kw)
            raw = quiet(build_sample_index, str(corpus.resolve()), **kw)
            assert torch.equal(dyn.window_session_ids, raw.window_session_ids)
            quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)

            pop = population(raw, K)
            if len(pop) < 2:
                print(f"  {name}: seed {seed} population too small at k={K} ({len(users)} users)")
                continue
            sizes.append(len(pop))
            pos = raw.samples[:, 4:7, :].mean(dim=2).numpy()
            emb = embed(model, dyn.samples)
            g_e, p_e, o = templates(emb, pop, K, np.random.default_rng(SEED))
            g_p, p_p, _ = templates(pos, pop, K, np.random.default_rng(SEED))
            d_dyn, d_y = cosine(p_e, g_e), euclid(p_p, g_p, [1])
            for label, d in (("dyn", d_dyn), ("y", d_y), ("y+dyn", zscore(d_dyn) + zscore(d_y))):
                per_seed[label].append(rank1(d, o, N_SMALL, np.random.default_rng(SEED)))
        if not per_seed["dyn"]:
            continue
        results[name] = {"k": K, "users": sizes, **per_seed}
        print(f"  {name:<18} users {min(sizes)}-{max(sizes)}   "
              + "   ".join(f"{lab} {np.mean(v):.3f}+-{np.std(v):.3f}" for lab, v in per_seed.items()),
              flush=True)
    pathlib.Path("docs/acceptance/step6_indomain_dyn.json").write_text(json.dumps(results, indent=1),
                                                                      encoding="utf-8")

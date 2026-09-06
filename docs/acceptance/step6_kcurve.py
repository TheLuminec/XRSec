"""
How much of the in-domain dyn figure is the model, and how much is 80 seconds of enrolment?

k=16 at 5 s is 80 s of gallery and 80 s of probe. The published rank-1 this project keeps
comparing itself to uses a SINGLE 15 s window. So the k=16 number cannot be set beside it,
and this curve produces the one that can.

Population is fixed from k=16 and every k scores those same users - the k-curve lesson:
a curve whose population changes with k measures the population.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model")); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index                 # noqa: E402
from normalization import ChannelNormalizer            # noqa: E402
from utils import load_checkpoint                      # noqa: E402
from step6_seated_dyn import (DYN_93, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, euclid, cosine, rank1)
from step6_indomain_dyn import validation_users, CORPORA, K   # noqa: E402

KS = (1, 3, 4, 8, 16)

if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N={N_SMALL} is {1/N_SMALL:.4f}; population fixed at k={K}\n")
    out = {}
    for name, corpus in CORPORA.items():
        per_k = {k: {"dyn": [], "y": []} for k in KS}
        sizes = []
        for seed, path in sorted(DYN_93.items()):
            model, ck = load_checkpoint(path, DEVICE, 100, return_checkpoint=True)
            users = validation_users(ck["eval_split"], seed, corpus)
            if not users:
                continue
            kw = dict(sample_time=5, sample_rate=20, exclude_users=users, swap_data=True)
            dyn = quiet(build_sample_index, str(corpus.resolve()), encoding="dyn", **kw)
            raw = quiet(build_sample_index, str(corpus.resolve()), **kw)
            quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
            pop = population(raw, K)          # fixed from the widest k, shared by every row
            if len(pop) < 2:
                continue
            sizes.append(len(pop))
            pos = raw.samples[:, 4:7, :].mean(dim=2).numpy()
            emb = embed(model, dyn.samples)
            for k in KS:
                g_e, p_e, o = templates(emb, pop, k, np.random.default_rng(SEED))
                g_p, p_p, _ = templates(pos, pop, k, np.random.default_rng(SEED))
                per_k[k]["dyn"].append(rank1(cosine(p_e, g_e), o, N_SMALL, np.random.default_rng(SEED)))
                per_k[k]["y"].append(rank1(euclid(p_p, g_p, [1]), o, N_SMALL, np.random.default_rng(SEED)))
        if not sizes:
            continue
        out[name] = {"users": sizes, **{str(k): v for k, v in per_k.items()}}
        print(f"  {name}  users {min(sizes)}-{max(sizes)}")
        for k in KS:
            d, y = per_k[k]["dyn"], per_k[k]["y"]
            print(f"    k={k:<3} ({k*5:>2}s enrolment)   dyn {np.mean(d):.3f}+-{np.std(d):.3f}"
                  f"   height {np.mean(y):.3f}+-{np.std(y):.3f}", flush=True)
    pathlib.Path("docs/acceptance/step6_kcurve_full.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

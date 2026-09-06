"""
The offset is real. Is it "a minority of very separable players", or is everyone above the
implication?

Those are different claims and only the first changes how 0.862 should be described. The
surviving offset is CONSISTENT with a minority - a heavy right tail pulls rank-1 above what
a Gaussian fitted to the whole score set predicts - but it is equally consistent with a
score distribution that is merely narrower than Gaussian for everybody. So this measures the
shape instead of inferring it.

Under the equal-variance Gaussian model every user is identical, so per-user rank-1 would be
concentrated at the implied value with nothing but the spread of a finite number of gallery
draws around it. The test is therefore not "is the mean higher" - we know it is - but "is the
per-user distribution wider than the draw noise, and where does the excess sit".

The null is simulated rather than argued: the same number of users, probes and draws, scored
from the fitted Gaussian, which gives the spread to beat.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import norm

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index                 # noqa: E402
from normalization import ChannelNormalizer            # noqa: E402
from utils import load_checkpoint                      # noqa: E402
from step6_seated_dyn import (DYN_93, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, cosine)
from step6_clean_boxrr import BOXRR, unseen_pool, K_POP  # noqa: E402
from step6_implied_rank1 import auc_from, implied_rank1  # noqa: E402

DRAWS = 300


def per_user_rank1(d, owners, n_gallery, rng):
    """Rank-1 for each enrolled user separately, averaged over that user's own probes and
    over random N-user galleries that always contain them."""
    n_users = d.shape[1]
    hits = np.zeros(n_users)
    counts = np.zeros(n_users)
    idx = np.arange(n_users)
    for _ in range(DRAWS):
        for i in range(len(d)):
            u = owners[i]
            others = rng.choice(np.delete(idx, u), n_gallery - 1, replace=False)
            row = d[i, np.append(others, u)]
            correct = d[i, u]
            better = int((row < correct).sum())
            tied = int((row == correct).sum())
            hits[u] += (1.0 / max(better + tied, 1)) if better == 0 else 0.0
            counts[u] += 1
    return hits / np.maximum(counts, 1)


def gaussian_null(auc, n_users, probes_per_user, rng):
    """Per-user rank-1 under the fitted equal-variance Gaussian, where every user is by
    construction identical. Whatever spread this shows is draw noise, not separability."""
    dprime = np.sqrt(2.0) * norm.ppf(auc)
    hits = np.zeros(n_users)
    for u in range(n_users):
        got = 0.0
        for _ in range(probes_per_user * DRAWS):
            genuine = rng.normal(dprime, 1.0)
            impostors = rng.normal(0.0, 1.0, N_SMALL - 1)
            got += float(genuine > impostors.max())
        hits[u] = got / (probes_per_user * DRAWS)
    return hits


if __name__ == "__main__":
    users = unseen_pool()
    kw = dict(sample_time=5, sample_rate=20, exclude_users=users, swap_data=True)
    raw = quiet(build_sample_index, str(BOXRR.resolve()), **kw)
    pop = population(raw, K_POP)
    print(f"population {len(pop)} users\n")

    per_user, aucs = [], []
    for seed, path in sorted(DYN_93.items()):
        model, ck = load_checkpoint(path, DEVICE, 100, return_checkpoint=True)
        dyn = quiet(build_sample_index, str(BOXRR.resolve()), encoding="dyn", **kw)
        quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
        emb = embed(model, dyn.samples)
        g_e, p_e, o = templates(emb, pop, 1, np.random.default_rng(SEED))
        d = cosine(p_e, g_e)
        mask = np.zeros_like(d, dtype=bool)
        mask[np.arange(len(d)), o] = True
        aucs.append(auc_from(-d[mask], -d[~mask]))
        per_user.append(per_user_rank1(d, o, N_SMALL, np.random.default_rng(SEED)))

    obs = np.mean(per_user, axis=0)          # average each user across the five checkpoints
    auc = float(np.mean(aucs))
    null = gaussian_null(auc, len(obs), 5, np.random.default_rng(SEED))

    print(f"AUC {auc:.4f}  implied rank-1 {implied_rank1(auc):.3f}  measured {obs.mean():.3f}\n")
    print(f'{"":<26}{"measured":>10}{"Gaussian null":>15}')
    print(f'{"mean":<26}{obs.mean():>10.3f}{null.mean():>15.3f}')
    print(f'{"sd across users":<26}{obs.std():>10.3f}{null.std():>15.3f}')
    for q in (10, 25, 50, 75, 90):
        print(f'{"p" + str(q):<26}{np.percentile(obs, q):>10.3f}{np.percentile(null, q):>15.3f}')
    print(f'{"users above 0.80":<26}{int((obs > 0.8).sum()):>10}{int((null > 0.8).sum()):>15}')
    print(f'{"users below 0.10":<26}{int((obs < 0.1).sum()):>10}{int((null < 0.1).sum()):>15}')

    top = np.sort(obs)[::-1]
    share = top.cumsum() / obs.sum()
    print(f"\ntop 10% of users carry {share[int(0.1*len(obs)) - 1]:.1%} of all correct identifications"
          f"  (Gaussian null: {np.sort(null)[::-1].cumsum()[int(0.1*len(null)) - 1] / null.sum():.1%})")
    print(f"top 25% carry {share[int(0.25*len(obs)) - 1]:.1%}"
          f"  (null {np.sort(null)[::-1].cumsum()[int(0.25*len(null)) - 1] / null.sum():.1%})")

    pathlib.Path("docs/acceptance/step6_separability_shape.json").write_text(
        json.dumps({"auc": auc, "measured_mean": float(obs.mean()), "implied": implied_rank1(auc),
                    "per_user": obs.tolist(), "null": null.tolist()}, indent=1), encoding="utf-8")

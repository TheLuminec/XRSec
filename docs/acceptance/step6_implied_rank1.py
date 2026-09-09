"""
Does BOXRR's rank-1 exceed what its own verification AUC implies - on ONE population?

CLAUDE.md's rule: compute what a verification number implies before running the
identification one, because a rank-1 that lands on the implication was already known and one
that lands more than ~0.05 away is the interesting case - a score distribution far from
Gaussian, i.e. a minority of very separable users.

The lead was AUC 0.814 implying 0.331 against a measured 0.449, but those are different
populations, so the offset could be population luck. This removes that entirely: the SAME
distance matrix that produces the rank-1 also contains every genuine and impostor score, so
verification and identification here are two readings of one score set on one set of users.
Nothing about the comparison is approximate except the Gaussian model itself, which is the
thing being tested.

The implication formula is gated against the three alyx per-axis values already published in
CLAUDE.md (0.593/0.661/0.539 -> 0.103/0.149/0.075) before it is used on anything new.

Registered by the Coordinator before this ran: the offset survives at +0.06 or more;
falsifier, under +0.02 and the 0.449 was population luck.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index                 # noqa: E402
from normalization import ChannelNormalizer            # noqa: E402
from utils import load_checkpoint                      # noqa: E402
from score_nymeria import gate                         # noqa: E402
from step6_seated_dyn import (write_gate_certificate, DYN_93, DEVICE, N_SMALL, SEED, quiet, embed, population,  # noqa: E402
                              templates, cosine, rank1)
from step6_clean_boxrr import BOXRR, unseen_pool, K_POP  # noqa: E402

KS = (1, 16)
ALYX_GATE = ((0.593, 0.103), (0.661, 0.149), (0.539, 0.075))   # published, N=17


def implied_rank1(auc: float, n: int = N_SMALL) -> float:
    """Rank-1 at gallery n under the equal-variance Gaussian score model.
    d' = sqrt(2) * Phi^-1(AUC); genuine ~ N(d',1), impostor ~ N(0,1)."""
    d = np.sqrt(2.0) * norm.ppf(auc)
    return float(quad(lambda x: norm.pdf(x - d) * norm.cdf(x) ** (n - 1), -12, 12, limit=200)[0])


def auc_from(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Rank-averaged AUC. Ties matter: a constant scorer must read 0.5."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s, i = scores[order], 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels == 1
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


if __name__ == "__main__":
    for a, t in ALYX_GATE:
        got = implied_rank1(a)
        assert abs(got - t) < 0.002, f"implication formula off: AUC {a} -> {got:.3f}, published {t}"
    print(f"implication formula gate: 3/3 against CLAUDE.md's alyx values\n")

    gates = [dict(gate(p), seed=s) for s, p in sorted(DYN_93.items())]
    write_gate_certificate(pathlib.Path(__file__).stem, gates)
    if not all(g["passed"] for g in gates):
        print("\n*** GATE FAILED ***")
        sys.exit(1)
    print("\ncheckpoint gate: 5/5 passed\n")

    users = unseen_pool()
    kw = dict(sample_time=5, sample_rate=20, exclude_users=users, swap_data=True)
    raw = quiet(build_sample_index, str(BOXRR.resolve()), **kw)
    pop = population(raw, K_POP)
    print(f"population {len(pop)} users; chance at N={N_SMALL} is {1/N_SMALL:.4f}\n")

    out = {k: {"auc": [], "measured": [], "implied": [], "offset": []} for k in KS}
    for seed, path in sorted(DYN_93.items()):
        model, ck = load_checkpoint(path, DEVICE, 100, return_checkpoint=True)
        dyn = quiet(build_sample_index, str(BOXRR.resolve()), encoding="dyn", **kw)
        quiet(ChannelNormalizer.from_state(ck.get("normalizer")).transform, dyn)
        emb = embed(model, dyn.samples)
        for k in KS:
            g_e, p_e, o = templates(emb, pop, k, np.random.default_rng(SEED))
            d = cosine(p_e, g_e)
            # One score set, two readings: the genuine cell of each row and every other cell.
            mask = np.zeros_like(d, dtype=bool)
            mask[np.arange(len(d)), o] = True
            auc = auc_from(-d[mask], -d[~mask])
            measured = rank1(d, o, N_SMALL, np.random.default_rng(SEED))
            imp = implied_rank1(auc)
            for key, v in (("auc", auc), ("measured", measured), ("implied", imp),
                           ("offset", measured - imp)):
                out[k][key].append(v)

    for k in KS:
        r = out[k]
        print(f"  k={k:<3} ({k*5:>2}s)   AUC {np.mean(r['auc']):.4f}   "
              f"implied {np.mean(r['implied']):.3f}   measured {np.mean(r['measured']):.3f}   "
              f"offset {np.mean(r['offset']):+.3f} +-{np.std(r['offset']):.3f}", flush=True)
    verdict = np.mean(out[1]["offset"])
    print(f"\nregistered: offset >= +0.06 survives, < +0.02 falsifies. k=1 offset {verdict:+.3f} -> "
          + ("SURVIVES" if verdict >= 0.06 else "FALSIFIED" if verdict < 0.02 else "BETWEEN - neither"))
    pathlib.Path("docs/acceptance/step6_implied_rank1.json").write_text(
        json.dumps({"users": len(pop), **{str(k): v for k, v in out.items()}}, indent=1), encoding="utf-8")

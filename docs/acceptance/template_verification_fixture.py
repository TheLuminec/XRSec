"""Fixture for template_verification's scoring core, answers known by construction (no checkpoint, no torch).
    python docs/acceptance/template_verification_fixture.py"""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from template_verification import cell_scores, summarise
rng = np.random.default_rng(0)
n, w, d = 30, 40, 16
centres = rng.normal(size=(n, d)) * 10
def corpus(sep):
    emb = np.concatenate([centres[u] * sep + rng.normal(size=(w, d)) for u in range(n)])
    rows = [np.arange(u * w, (u + 1) * w) for u in range(n)]
    return emb, [r[: w // 2] for r in rows], [r[w // 2:] for r in rows]
emb, g, p = corpus(1.0)                      # perfectly separable
s = summarise([cell_scores(emb, g, p)])
assert s["eer"] == 0, s["eer"]
# cross-fitted: the threshold comes from the other fold's gap, so a held-out fold can sit a hair over it
assert s["crossfit_eer_threshold"]["frr"] < 0.005 and s["crossfit_eer_threshold"]["far"] < 0.005, s["crossfit_eer_threshold"]
assert all(x == 1 for x in s["rank1_per_user"]) and s["crossfit_far1pct"]["tar"] > 0.995, s["crossfit_far1pct"]
emb, g, p = corpus(0.0)                      # no identity at all
s = summarise([cell_scores(emb, g, p), cell_scores(emb, p, g)])
assert 0.45 < s["eer"] < 0.55, s["eer"]
assert 0.45 < s["crossfit_eer_threshold"]["balanced_accuracy"] < 0.55, s["crossfit_eer_threshold"]
assert s["crossfit_far1pct"]["tar"] < 0.03 and s["crossfit_far1pct"]["realised_far"] < 0.03, s["crossfit_far1pct"]
assert abs(np.mean(s["rank1_per_user"]) - 1 / n) < 0.03
assert len(s["per_user_frr"]) == n and s["impostor"] == s["genuine"] * (n - 1)
print("FIXTURE PASSES: separable -> EER 0, cross-fitted FAR/FRR < 0.005, TAR@1% > 0.995, rank-1 1; identity-free -> EER ~0.5, balanced accuracy ~0.5, "
      "TAR@1% ~ FAR, rank-1 ~ 1/N; impostor count = genuine x (N-1)")

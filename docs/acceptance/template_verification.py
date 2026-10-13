"""
Template verification: the deployment decision Dr. Feng asked for (email 2026-10-12, point 2), computed on
the same embeddings our rank-1 figures use. Protocol: template_verification_PROTOCOL.md, committed with this
file before any checkpoint was scored with it.

A user enrols a template: the renormalised mean of their L2-normalised enrolment-window embeddings, the same
`centroids` as A1. Each probe window is compared with a claimed template by cosine. A genuine attempt is a
probe against its own user's template; an impostor attempt is a probe against another user's template. The
accept/reject threshold is never fitted on the people it is scored on: users are split into two folds at
random, the threshold is fitted on one fold's attempts (impostors drawn only from inside that fold) and
applied to the other, then swapped, over R repeats (seed 67).

Reported per condition:
  eer                      equal error rate on all users' attempts (threshold-free, like AUC)
  crossfit_eer_threshold   FAR, FRR and balanced accuracy at a threshold fitted at the EER on the other fold
  crossfit_far1pct         TAR and the realised FAR at a threshold fitted for FAR = 1 % on the other fold
  per_user_frr             each user's FRR at the cross-fitted EER threshold, averaged over the repeats in
                           which that user was held out (the distribution the mean hides)
  rank1_per_user           A1 rank-1 from the same score matrix, as a plumbing gate against committed figures

Adapters, each run on the device the checkpoint's committed figures came from, gated first:
  axr           Across-XR users 32-48: cross-application (20 ordered cells) and within-application (A0)
  ballthrowing  41 people: C0 same session, C1 same headset other day, C2 other headset
  alyx          the Nymeria-in-domain seed's alyx validation users with two dates: cross-day and same-day

    python docs/acceptance/template_verification.py axr --checkpoints SEED=ARM=path [...] --device cuda --out DIR \
        [--committed SEED=ARM=committed.json ...]
    python docs/acceptance/template_verification.py ballthrowing ...
    python docs/acceptance/template_verification.py alyx ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
from across_xr_alignment import centroids, l2, rank1_per_user  # noqa: E402

REPEATS = 20
SEED = 67
FAR_TARGET = 0.01


# ------------------------------------------------------------------ scoring core (no torch needed)

def cell_scores(emb: np.ndarray, gal_rows: list, probe_rows: list) -> dict:
    """One cell: templates from gal_rows (one per user), probes from probe_rows. Returns the full
    probe x template cosine matrix and each probe's own user."""
    n = len(gal_rows)
    assert len(probe_rows) == n
    templates = centroids(emb, gal_rows)
    pu = np.concatenate([np.full(len(r), i) for i, r in enumerate(probe_rows)])
    S = l2(emb[np.concatenate(probe_rows)]) @ templates.T
    return {"S": S, "pu": pu, "n": n, "rank1": rank1_per_user(templates, emb[np.concatenate(probe_rows)], pu, n)}


def _attempts(cells: list[dict], users: np.ndarray):
    """Genuine scores (with user) and impostor scores restricted to templates of `users`."""
    keep = np.zeros(cells[0]["n"], bool); keep[users] = True
    gen, gen_u, imp = [], [], []
    for c in cells:
        m = keep[c["pu"]]
        S, pu = c["S"][m], c["pu"][m]
        gen.append(S[np.arange(len(pu)), pu]); gen_u.append(pu)
        other = np.where(keep)[0]
        I = S[:, other]
        own = (other[None, :] == pu[:, None])
        imp.append(I[~own])
    return np.concatenate(gen), np.concatenate(gen_u), np.concatenate(imp)


def eer_threshold(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """Threshold t (accept if score >= t) minimising |FAR - FRR|; returns (t, EER)."""
    # every genuine score is a candidate (the separating threshold of a perfect scorer is one), plus a
    # 4,001-point quantile grid of the impostors, so large impostor sets stay cheap
    cand = np.unique(np.concatenate([gen, np.quantile(imp, np.linspace(0, 1, 4001))]))
    g, i = np.sort(gen), np.sort(imp)
    frr = np.searchsorted(g, cand, side="left") / len(g)
    far = 1 - np.searchsorted(i, cand, side="left") / len(i)
    k = int(np.argmin(np.abs(far - frr)))
    # place the threshold midway into the gap below the chosen candidate, not on a genuine score, so a
    # held-out fold is not penalised for the fitting fold's exact minimum (matters for separable data)
    t = float((cand[k - 1] + cand[k]) / 2) if k > 0 else float(cand[0])
    return t, float(((imp >= t).mean() + (gen < t).mean()) / 2)


def far_threshold(imp: np.ndarray, target: float) -> float:
    """Smallest threshold whose FAR on these impostors is at most `target`."""
    return float(np.quantile(imp, 1 - target, method="higher"))


def summarise(cells: list[dict]) -> dict:
    n = cells[0]["n"]
    assert all(c["n"] == n for c in cells)
    allu = np.arange(n)
    gen, gen_u, imp = _attempts(cells, allu)
    _, eer = eer_threshold(gen, imp)
    rng = np.random.default_rng(SEED)
    rows, frr_u, far1 = [], [[] for _ in range(n)], []
    for _ in range(REPEATS):
        perm = rng.permutation(n)
        folds = (np.sort(perm[: n // 2]), np.sort(perm[n // 2:]))
        for fit, test in (folds, folds[::-1]):
            fg, _, fi = _attempts(cells, fit)
            t, _ = eer_threshold(fg, fi)
            tg, tu, ti = _attempts(cells, test)
            far, frr = float((ti >= t).mean()), float((tg < t).mean())
            rows.append((far, frr))
            for u in test:
                frr_u[u].append(float((tg[tu == u] < t).mean()))
            t1 = far_threshold(fi, FAR_TARGET)
            far1.append((float((tg >= t1).mean()), float((ti >= t1).mean())))
    far_m, frr_m = np.mean(rows, axis=0)
    per_user = np.array([np.mean(v) for v in frr_u])
    return {"n_users": int(n), "cells": len(cells), "genuine": int(len(gen)), "impostor": int(len(imp)),
            "eer": eer,
            "crossfit_eer_threshold": {"far": float(far_m), "frr": float(frr_m),
                                       "balanced_accuracy": float(1 - (far_m + frr_m) / 2), "repeats": REPEATS},
            "crossfit_far1pct": {"tar": float(np.mean([a for a, _ in far1])), "realised_far": float(np.mean([b for _, b in far1]))},
            "per_user_frr": per_user.tolist(),
            "per_user_frr_quantiles": {q: float(np.quantile(per_user, float(q))) for q in ("0.1", "0.5", "0.9")},
            "rank1_per_user": np.mean([c["rank1"] for c in cells], axis=0).tolist()}


# ------------------------------------------------------------------ adapters

def _gate(path, device, tol):
    from across_xr_alignment import gate
    g, model, ck = gate(path, device)
    ok = bool(g.get("passed")) and g.get("gap") is not None and g["gap"] <= tol
    return ok, g, model, ck


def adapter_axr(path, device, tol):
    from across_xr_alignment import Corpus, cross_cells
    ok, g, model, ck = _gate(path, device, tol)
    if not ok:
        return {"refused": f"gate {g.get('gap')} > {tol} ({g.get('reason', '')})"}
    corpus = Corpus(ck, device, model)
    test = corpus.split["test"]
    assert len(test) == 17
    emb = corpus.embeddings
    cross = [cell_scores(emb, corpus.rows(test, a), corpus.rows(test, b)) for a, b in cross_cells()]
    games = sorted({a for a, _ in cross_cells()})
    within = [cell_scores(emb, corpus.rows(test, a, half="first"), corpus.rows(test, a, half="second")) for a in games]
    return {"gate": {k: v for k, v in g.items() if k != "exclude_users_seen"}, "users": [int(u) for u in test],
            "conditions": {"cross_application": summarise(cross), "within_application": summarise(within)}}


def adapter_ballthrowing(path, device, tol):
    from ballthrowing_cross_day import C1_PAIRS, C2_PAIRS, SESSIONS, load_corpus, rows
    ok, g, model, ck = _gate(path, device, tol)
    if not ok:
        return {"refused": f"gate {g.get('gap')} > {tol} ({g.get('reason', '')})"}
    emb, _, users, where, _ = load_corpus(ck, model, device)
    U = range(len(users))
    c0 = []
    for s in SESSIONS:
        a = [rows(where, u, s, range(0, 5)) for u in U]; b = [rows(where, u, s, range(5, 10)) for u in U]
        c0 += [cell_scores(emb, a, b), cell_scores(emb, b, a)]
    def both(pairs):
        out = []
        for x, y in pairs:
            gx = [rows(where, u, x) for u in U]; gy = [rows(where, u, y) for u in U]
            out += [cell_scores(emb, gx, gy), cell_scores(emb, gy, gx)]
        return out
    return {"gate": {k: v for k, v in g.items() if k != "exclude_users_seen"}, "users": users,
            "conditions": {"C0_same_session": summarise(c0), "C1_same_headset_other_day": summarise(both(C1_PAIRS)),
                           "C2_other_headset": summarise(both(C2_PAIRS))}}


def adapter_alyx(path, device, tol, seed):
    """Rows exactly as alyx_cross_day.score_checkpoint builds them (gated against its committed per-user rank-1)."""
    from alyx_cross_day import ALYX, eligible_users
    from across_xr_alignment import embed, quiet
    from dataset import SampleDataset, SampleIndex
    from normalization import ChannelNormalizer
    ok, g, model, ck = _gate(path, device, tol)
    if not ok:
        return {"refused": f"gate {g.get('gap')} > {tol} ({g.get('reason', '')})"}
    es = ck["eval_split"]
    users = eligible_users(seed)
    ds = quiet(SampleDataset, str(ALYX), sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
               channels=ck.get("channels", "full"), resample=es.get("resample", "nearest"),
               window_stride=es.get("window_stride"), exclude_users=users, swap_data=True)
    assert sorted(ds.user_dirs) == users
    index = SampleIndex(ds, encoding=es.get("encoding", "raw"))
    norm = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if norm.enabled:
        quiet(norm.transform, index)
    assert not norm.unseen_datasets
    emb = embed(model, index.samples, device)
    sess, start, T = index.window_session_ids.numpy(), index.window_start_times.numpy().astype(float), float(es["sample_time"])
    rws = []
    for r in index.user_sample_indices:
        r = r.numpy(); s0, s1 = r[sess[r] == 0], r[sess[r] == 1]
        halves = []
        for s in (s0, s1):
            cut = np.median(start[s]); halves.append((s[start[s] + T <= cut], s[start[s] >= cut]))
        rws.append({"s0": s0, "s1": s1, "halves": halves})
    cross = [cell_scores(emb, [x["s0"] for x in rws], [x["s1"] for x in rws]),
             cell_scores(emb, [x["s1"] for x in rws], [x["s0"] for x in rws])]
    same = [cell_scores(emb, [x["halves"][k][0] for x in rws], [x["halves"][k][1] for x in rws]) for k in (0, 1)]
    names = [pathlib.Path(u).name for u in ds.user_dirs]
    return {"gate": {k: v for k, v in g.items() if k != "exclude_users_seen"}, "users": names,
            "conditions": {"cross_day": summarise(cross), "same_day": summarise(same)}}


def plumbing_check(res: dict, committed: dict | None, corpus: str) -> dict:
    """The rank-1 computed from the same score matrices must equal the committed figures exactly (same
    device and checkpoint). Without that, nothing here is read."""
    if committed is None:
        return {"checked": False}
    c = res["conditions"]
    if corpus == "axr":
        ours = np.array(c["cross_application"]["rank1_per_user"])
        theirs = np.asarray(committed["arms"]["A1"]["per_user"], float)
    elif corpus == "ballthrowing":
        ours = np.array(c["C1_same_headset_other_day"]["rank1_per_user"])
        theirs = np.array([committed["C1"][u] for u in res["users"]])
    else:
        ours = np.array(c["cross_day"]["rank1_per_user"])
        theirs = np.array([committed["cross_day"][u] for u in res["users"]])
    gap = float(np.max(np.abs(ours - theirs)))
    return {"checked": True, "max_abs_gap": gap, "passed": gap < 1e-6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=("axr", "ballthrowing", "alyx"))
    ap.add_argument("--checkpoints", nargs="+", required=True, help="SEED=ARM=path")
    ap.add_argument("--committed", nargs="*", default=[], help="SEED=ARM=committed result json, for the plumbing gate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "docs" / "acceptance"))
    a = ap.parse_args()
    import torch
    device = torch.device(a.device)
    tol = {"cpu": 1e-3, "cuda": 1e-4}[device.type]
    committed = {}
    for item in a.committed:
        s, arm, p = item.split("=", 2)
        d = json.loads(pathlib.Path(p).read_text())
        if a.corpus == "axr":
            d = d["seeds"][0]
        else:
            d = [r for r in d["results"] if "refused" not in r and int(r.get("seed", -1)) == int(s) and r.get("arm", arm) == arm][0]
        committed[(int(s), arm)] = d
    rc = 0
    for item in a.checkpoints:
        s, arm, path = item.split("=", 2)
        fn = {"axr": adapter_axr, "ballthrowing": adapter_ballthrowing}.get(a.corpus)
        res = fn(path, device, tol) if fn else adapter_alyx(path, device, tol, int(s))
        res.update(checkpoint=path, seed=int(s), arm=arm, device=str(device), protocol="docs/acceptance/template_verification_PROTOCOL.md")
        if "refused" not in res:
            res["plumbing"] = plumbing_check(res, committed.get((int(s), arm)), a.corpus)
            if res["plumbing"].get("checked") and not res["plumbing"]["passed"]:
                res = {**{k: res[k] for k in ("checkpoint", "seed", "arm", "device", "plumbing")},
                       "refused": f"plumbing: rank-1 differs from committed by {res['plumbing']['max_abs_gap']:.2e}"}
        out = pathlib.Path(a.out) / f"tv_{a.corpus}_{arm}_s{s}_{device.type}.json"
        out.write_text(json.dumps(res, indent=1))          # written before anything is read from it
        if "refused" in res:
            rc = 1; print(f"REFUSED {a.corpus} {arm} s{s}: {res['refused']}")
        else:
            print(f"{a.corpus} {arm} s{s}: gate {res['gate'].get('gap'):.1e} plumbing {res['plumbing']} -> {out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

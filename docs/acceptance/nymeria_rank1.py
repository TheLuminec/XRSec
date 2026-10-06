"""
Rank-1 identification for the Nymeria in-domain arm (nymeria_rank1_REGISTERED.md), so the AR-glasses
row sits on the same axis (rank-1 at N=17) as every other identification row in the paper.

Population: the 48 held-out Nymeria users (nymeria_in_domain_heldout48.txt), never trained or validated
on by either arm. Checkpoints: the six 120-epoch checkpoints of nymeria_in_domain_REGISTERED.md
(3 seeds x {treatment, control}). Each is GATED FIRST through across_xr_alignment.gate (its own
selected_test_auc on its own evaluation users, tolerance cuda 1e-4 / cpu 1e-3) and nothing is read from
a checkpoint that fails. Embeddings: the pipeline's own SampleDataset / SampleIndex / the checkpoint's
normaliser on exactly those 48 users, `dyn` 10 s stride 5; 47,796 windows asserted.

A1 rule throughout (imported, one implementation): L2-normalised window embeddings, a template is the
renormalised centroid of a user's windows, cosine, ties rank-averaged so a tie is never rank 1
(`centroids` / `rank1_per_user` from across_xr_alignment; the full-N computation here is asserted equal
to `rank1_per_user` on every group at run time). Probe unit: ONE WINDOW (10 s), as in A1.

PROTOCOLS
---------
constrained  (PRIMARY; "script-matched gallery, probe script excluded from every template")
    For each script s: the candidates are the held-out users who recorded s; every candidate's template
    is the centroid of that candidate's windows from scripts OTHER THAN s; the probes are the candidates'
    s-windows. So the true person is always enrolled on activities other than the probe's, and so is
    every impostor.
    Why this and not the alternatives (decided before any number):
      * the preferred per-pair design (gallery script g, probe script p, all templates from g) needs
        users with both g and p; on the 48 only 4 ordered pairs reach 17 users and 20 reach 10
        (nymeria_sequence_scripts.csv x heldout48), covering 72 of 700 user-pair slots at N>=17 -
        too thin for the primary. It is kept as the secondary "pair" protocol below.
      * the plain fallback (all 48 users as candidates, every template minus script s) does NOT deny the
        activity cue on this corpus: Nymeria scripts come in bundles (co-occurrence lift up to 6.3 over
        the 188 non-held-out participants, several pairs never co-occur), so the true person - who by
        construction recorded s - shares s's bundle more often than a random impostor, and a template of
        the person's OTHER scripts still carries "which kind of activities they did". Restricting the
        candidates to people who ALSO recorded s makes the genuine user exchangeable with every impostor
        under ANY activity-only embedding: chance in expectation, whatever the bundle structure. The
        fixture measures this on the real 48-user script table, both protocols, both ways.
      * this mirrors the verification constrained protocol's logic (activity matched between genuine and
        impostor) without its adversarial half: there, negatives are SAME-script pairs, which pushes an
        activity cue below chance (the zero-shot control's 0.47); here the probe's script is absent from
        every template, so an activity cue is neutral, not reversed. It also mirrors Across-XR A1, where
        every candidate recorded the probe's application and templates come from a different one.
    N=17: for every probe, DRAWS galleries of 16 impostors drawn from the other candidates of its
    script, seed DRAW_SEED, identical draws for every checkpoint (paired). Only scripts with >= 17
    candidates enter N=17 (5 scripts on the 48); users with no such script have no N=17 value.
    N=all: every script with >= 2 candidates, at its own N; chance is reported per user beside it.
unconstrained  (comparison; "leave the probe's sequence out, any script")
    Candidates: all 48. Templates: every impostor's centroid over all their windows; the true person's
    over all windows except the probe's own sequence (no shared frames). The identification analogue of
    the row AUC (cross-recording positives, any script). Not activity-neutral in either direction.
pair  (secondary, exact denial)
    Ordered (g, p), g != p, users with both, N >= MIN_PAIR_N: templates from g only, probes from p.
    Reported per cell at N = eligible users (chance 1/N), and pooled per user.
fallback_all48  (diagnostic only; the rejected fallback)
    As constrained, but every held-out user is a candidate whether or not they recorded s. Reported so
    the size of the bundle leak it admits can be read off the real model, beside the fixture's measure
    of it under an activity-only embedding. Never quoted as an identification figure.

THE REGISTERED STATISTIC IS CELL-BALANCED, and why (measured in the fixture before any number existed).
A cell is one (user, gallery) - for constrained a (user, probe script) pair; its value is the hit rate
over that user's probes in that gallery; the figure is the mean over cells, bootstrapped over users. Under
an activity-only embedding a probe's embedding does not depend on its owner, so within one gallery the
candidates' cell rates sum to 1 at N=all and to n/17 at N=17 (sum_k C(n-k,16)/C(n-1,16) = n/17): the
cell mean is chance in expectation whatever the activity geometry and window counts. The two obvious
alternatives are not: the per-user mean (each user's probes pooled) reweights cells by how many
eligible scripts a user has, and on the real 48-user table one activity-only embedding moved it up to
0.057 from 1/17 (fixture 5; ACTIVITY_FLOOR) - as large as the control band. The cell-balanced mean
stayed within 0.01 in every realization. Per-user and probe-pooled means are reported beside it.
fallback_all48 breaks the argument: candidates who did NOT record s own no probes, so if bundle
structure concentrates the argmax on people who did, the cell mean exceeds 1/48 (fixture 5: +0.05 to
+0.08 at N=17 under a bundle-aligned activity embedding).

Each checkpoint also reports, on the SAME score set as its N=17 rank-1, the verification AUC
(genuine = probe vs own template, impostor = probe vs every other candidate's template) and the rank-1
it implies under the equal-variance Gaussian model (CLAUDE.md: "a pairwise AUC implies a rank-1"; the
formula is gated in the fixture against the published alyx triple).

    # score (Miami, cuda) - one JSON per checkpoint, written the moment it is scored
    python docs/acceptance/nymeria_rank1.py score --device cuda --out-dir docs/acceptance \
        --checkpoints 1=control=runs/2026-09-21/07-55-55_train/checkpoints/<stem>.pth [...]
    # read (anywhere, committed JSONs only)
    python docs/acceptance/nymeria_rank1.py read docs/acceptance/nymeria_rank1_*_s*_cuda.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
from across_xr_alignment import N_BOOT, centroids, embed, gate, l2, quiet, rank1_per_user  # noqa: E402
from exposure_breadth_read import ci  # noqa: E402

NYM = ROOT / "processed_datasets" / "Nymeria_Dataset" / "users"
HELD = ROOT / "docs" / "acceptance" / "nymeria_in_domain_heldout48.txt"
GATE_TOL = {"cpu": 1e-3, "cuda": 1e-4}
EXPECTED_USERS, EXPECTED_WINDOWS = 48, 47796      # nymeria_script_pair.json, every 120-epoch checkpoint
N_SMALL, DRAWS, DRAW_SEED, MIN_PAIR_N = 17, 200, 67, 10
# Measured by nymeria_rank1_fixture.py (5) on the real 48-user script table: a single activity-only
# embedding moved the constrained PER-USER N=17 mean up to this far from 1/17 (worst realization). The
# registered CELL-BALANCED mean is not movable by activity alone (exact by construction; asserted there).
ACTIVITY_FLOOR = 0.057
READ_OUT = ROOT / "docs" / "acceptance" / "nymeria_rank1_read.json"
PROTOCOLS = ("constrained", "unconstrained", "pair", "fallback_all48")


def load_heldout() -> list[str]:
    users = [l.strip() for l in HELD.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(users) == EXPECTED_USERS == len(set(users)), len(users)
    return sorted(users)


# --------------------------------------------------------------------------------------------------
# Pure numpy: protocols and scoring. This is what nymeria_rank1_fixture.py exercises.
# --------------------------------------------------------------------------------------------------

def groups_for(protocol: str, win_user: np.ndarray, win_seq: np.ndarray, win_script: np.ndarray) -> list[dict]:
    """Each group: probes (window rows), cand (user ids, the gallery), tmpl (one row array per candidate),
    label. The probe's true user is always in cand."""
    users = np.unique(win_user)
    rows_u = {u: np.flatnonzero(win_user == u) for u in users}
    scripts_u = {u: set(win_script[rows_u[u]].tolist()) for u in users}
    out = []
    if protocol == "constrained":
        for s in np.unique(win_script):
            cand = np.array([u for u in users if s in scripts_u[u] and len(scripts_u[u]) >= 2])
            if len(cand) < 2:
                continue
            tmpl = [rows_u[v][win_script[rows_u[v]] != s] for v in cand]
            probes = np.flatnonzero((win_script == s) & np.isin(win_user, cand))
            out.append({"label": int(s), "probes": probes, "cand": cand, "tmpl": tmpl})
    elif protocol == "unconstrained":
        for q in np.unique(win_seq):
            probes = np.flatnonzero(win_seq == q)
            u = win_user[probes[0]]
            rest = rows_u[u][win_seq[rows_u[u]] != q]
            if len(rest) == 0:
                continue
            tmpl = [rest if v == u else rows_u[v] for v in users]
            out.append({"label": int(q), "probes": probes, "cand": users, "tmpl": tmpl})
    elif protocol == "fallback_all48":
        for s in np.unique(win_script):
            cand = np.array([u for u in users if (scripts_u[u] - {s})])
            probes = np.flatnonzero((win_script == s) & np.isin(win_user, cand))
            if len(cand) < 2 or len(probes) == 0:
                continue
            tmpl = [rows_u[v][win_script[rows_u[v]] != s] for v in cand]
            out.append({"label": int(s), "probes": probes, "cand": cand, "tmpl": tmpl})
    elif protocol == "pair":
        for g in np.unique(win_script):
            for p in np.unique(win_script):
                if g == p:
                    continue
                cand = np.array([u for u in users if g in scripts_u[u] and p in scripts_u[u]])
                if len(cand) < MIN_PAIR_N:
                    continue
                tmpl = [rows_u[v][win_script[rows_u[v]] == g] for v in cand]
                probes = np.flatnonzero((win_script == p) & np.isin(win_user, cand))
                out.append({"label": [int(g), int(p)], "probes": probes, "cand": cand, "tmpl": tmpl})
    else:
        raise ValueError(protocol)
    for grp in out:   # no template may share a window with a probe of its own user (self-match guard)
        for v, t in zip(grp["cand"], grp["tmpl"]):
            assert len(t), f"{protocol}: empty template for user {v}"
            assert not np.isin(grp["probes"][win_user[grp["probes"]] == v], t).any(), f"{protocol}: probe in own template"
    return out


def auc_ranked(gen: np.ndarray, imp: np.ndarray) -> float:
    """Rank-averaged AUC; a constant scorer reads exactly 0.5."""
    s = np.concatenate([gen, imp])
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = np.empty(len(s))
    ranks[order] = avg[inv]
    ng, ni = len(gen), len(imp)
    return float((ranks[:ng].sum() - ng * (ng + 1) / 2.0) / (ng * ni))


def implied_rank1(auc: float, n: int) -> float:
    """Equal-variance Gaussian: d' = sqrt(2) Phi^-1(AUC); rank-1 = P(genuine beats n-1 impostors)."""
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < auc else (lo, mid)
    d = math.sqrt(2.0) * (lo + hi) / 2
    x = np.linspace(-12, 12, 48001)
    cdf = 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in x]))
    f = np.exp(-0.5 * (x - d) ** 2) / math.sqrt(2 * math.pi) * cdf ** (n - 1)
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(x)))


def score_protocol(emb: np.ndarray, win_user: np.ndarray, win_seq: np.ndarray, win_script: np.ndarray,
                   protocol: str, n_small: int = N_SMALL, draws: int = DRAWS, draw_seed: int = DRAW_SEED) -> dict:
    """Per-user rank-1 at N=all-candidates ('full') and at N=n_small by repeated gallery draws ('n17').
    Users are the integer ids in win_user; per-user arrays are indexed by them (nan = no probe)."""
    n_users = int(win_user.max()) + 1
    rng = np.random.default_rng(draw_seed)
    acc = {k: defaultdict(lambda: np.zeros(n_users)) for k in ("full", "n17")}
    ucell = {k: defaultdict(list) for k in ("full", "n17")}      # user -> [[group index, rate, chance], ...]
    gen_s, imp_s, cells = [], [], []
    for gi, grp in enumerate(groups_for(protocol, win_user, win_seq, win_script)):
        cand, probes = grp["cand"], grp["probes"]
        n = len(cand)
        pos = {int(v): k for k, v in enumerate(cand)}
        o = np.array([pos[int(u)] for u in win_user[probes]])
        T = centroids(emb, grp["tmpl"])
        S = l2(emb[probes]) @ l2(T).T                                   # rank1_per_user's exact expression
        own = S[np.arange(len(probes)), o]
        better = (S > own[:, None]).sum(axis=1)
        ties = (S == own[:, None]).sum(axis=1) - 1
        rank = 1 + better + 0.5 * ties
        hit = (rank <= 1.0).astype(float)
        mine = np.array([hit[o == k].mean() if (o == k).any() else np.nan for k in range(n)])
        ref = rank1_per_user(T, emb[probes], o, n)                       # the imported rule, asserted equal
        assert np.allclose(mine, ref, equal_nan=True, atol=0, rtol=0), f"{protocol} {grp['label']}: rank rule mismatch"
        users_p = cand[o]
        np.add.at(acc["full"]["hits"], users_p, hit)
        np.add.at(acc["full"]["count"], users_p, 1.0)
        np.add.at(acc["full"]["chance"], users_p, 1.0 / n)
        np.add.at(acc["full"]["rank"], users_p, rank)
        for k in np.unique(o):
            ucell["full"][int(cand[k])].append([gi, float(hit[o == k].mean()), 1.0 / n])
        cell = {"label": grp["label"], "n": n, "probes": int(len(probes)), "rank1_full": float(hit.mean())}
        if n >= n_small:
            geq = S >= own[:, None]
            geq[np.arange(len(probes)), o] = False
            h = np.zeros(len(probes))
            for _ in range(draws):
                keys = rng.random((n, n))
                np.fill_diagonal(keys, np.inf)
                imp = np.argsort(keys, axis=1)[:, : n_small - 1]          # 16 impostors per genuine candidate
                M = np.zeros((n, n), dtype=bool)
                M[np.arange(n)[:, None], imp] = True
                h += ~(geq & M[o]).any(axis=1)
            np.add.at(acc["n17"]["hits"], users_p, h / draws)
            np.add.at(acc["n17"]["count"], users_p, 1.0)
            np.add.at(acc["n17"]["chance"], users_p, 1.0 / n_small)
            for k in np.unique(o):
                ucell["n17"][int(cand[k])].append([gi, float((h / draws)[o == k].mean()), 1.0 / n_small])
            gen_s.append(own)
            imp_s.append(S[~np.eye(n, dtype=bool)[o]])
            cell["rank1_n17"] = float((h / draws).mean())
        cells.append(cell)
    out = {"cells": cells}
    for k, a in acc.items():
        cnt = a["count"]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_user = np.where(cnt > 0, a["hits"] / cnt, np.nan)
            chance = np.where(cnt > 0, a["chance"] / cnt, np.nan)
            mrank = np.where(cnt > 0, a["rank"] / cnt, np.nan) if k == "full" else None
        out[k] = {"per_user": per_user, "chance_per_user": chance, "n_probes": cnt,
                  "mean": float(np.nanmean(per_user)) if np.isfinite(per_user).any() else float("nan"),
                  "chance": float(np.nanmean(chance)) if np.isfinite(chance).any() else float("nan"),
                  "users": int(np.isfinite(per_user).sum()),
                  # probe-weighted: the statistic the activity-neutrality argument is exact for
                  "pooled": float(a["hits"].sum() / cnt.sum()) if cnt.sum() else float("nan"),
                  "pooled_chance": float(a["chance"].sum() / cnt.sum()) if cnt.sum() else float("nan")}
        if mrank is not None:
            out[k]["mean_rank_per_user"] = mrank
        # CELL-BALANCED (REGISTERED): the unit is one (user, gallery) cell - for constrained a (user, probe
        # script) pair - each cell the hit rate over that user's probes in that gallery; the figure is the
        # mean over cells. Under an activity-only embedding a probe's embedding does not depend on its
        # owner, so within one gallery the candidates' cell rates sum to exactly 1 at N=all and to n/17 at
        # N=17 (sum over ranks k of C(n-k,16)/C(n-1,16) = n/17), whatever the geometry and window counts:
        # the mean over cells is chance exactly. The per-user mean and the probe-pooled mean are not
        # (fixture 5 measures how far they move).
        allc = [c for u in sorted(ucell[k]) for c in ucell[k][u]]
        out[k]["cells_per_user"] = {u: v for u, v in ucell[k].items()}
        out[k]["cell_mean"] = float(np.mean([c[1] for c in allc])) if allc else float("nan")
        out[k]["cell_chance"] = float(np.mean([c[2] for c in allc])) if allc else float("nan")
        out[k]["n_cells"] = len(allc)
    if gen_s:
        a = auc_ranked(np.concatenate(gen_s), np.concatenate(imp_s))
        out["n17"]["same_score_auc"] = a
        out["n17"]["implied_rank1"] = implied_rank1(a, n_small)
    return out


def score_all(emb, win_user, win_seq, win_script, names: list[str]) -> dict:
    res = {}
    for p in PROTOCOLS:
        r = score_protocol(emb, win_user, win_seq, win_script, p)
        for k in ("full", "n17"):
            for key in ("per_user", "chance_per_user", "n_probes", "mean_rank_per_user"):
                if key in r[k]:
                    r[k][key] = {names[u]: float(v) for u, v in enumerate(r[k][key]) if np.isfinite(v) and (key != "n_probes" or v > 0)}
            r[k]["cells_per_user"] = {names[u]: v for u, v in r[k]["cells_per_user"].items()}
        res[p] = r
    return res


# --------------------------------------------------------------------------------------------------
# Checkpoint path (Miami)
# --------------------------------------------------------------------------------------------------

def index_metadata(index, table: dict, held: list[str]):
    """win_user / win_seq / win_script integer arrays from the pipeline's own index; script labels
    joined exactly as nymeria_script_pair does (participant, act) on its UserProfile file order."""
    from nymeria_script_pair import window_scripts
    names = [pathlib.Path(d).name for d in index.user_dirs]
    assert names == held, "index users are not the 48 held-out users in sorted order"
    scripts = window_scripts(index, table)
    assert all(s is not None for s in scripts), "a window has no script label"
    W = index.sample_count
    win_user = np.empty(W, dtype=np.int64)
    for u, idx in enumerate(index.user_sample_indices):
        win_user[idx.numpy()] = u
    sess = index.window_session_ids.numpy().astype(np.int64)
    win_seq = win_user * 1000 + sess
    vocab = sorted(set(scripts))
    win_script = np.array([vocab.index(s) for s in scripts], dtype=np.int64)
    return win_user, win_seq, win_script, vocab, names


def score_checkpoint(path: str, seed: int, arm: str, device) -> dict:
    from dataset import SampleDataset, SampleIndex
    from normalization import ChannelNormalizer
    from nymeria_script_pair import scripts_by_sequence
    g, model, ck = gate(path, device)
    tol = GATE_TOL[device.type]
    if not g.get("passed") or g.get("gap") is None or g["gap"] > tol:
        return {"checkpoint": path, "seed": seed, "arm": arm,
                "refused": f"gate gap {g.get('gap')} > {tol} on {device} ({g.get('reason', '')})"}
    held = load_heldout()
    es = ck["eval_split"]
    assert g["eval_users"] == EXPECTED_USERS, g["eval_users"]
    excl = sorted(pathlib.Path(str(u).replace("\\", "/")).name for u in (es.get("exclude_users") or [])
                  if "Nymeria_Dataset" in str(u))
    assert excl == held, "checkpoint's excluded Nymeria users are not heldout48"
    assert bool(es.get("test_on_excluded")), "checkpoint did not evaluate on its excluded users"
    seen = {pathlib.Path(str(v).replace("\\", "/")).name for v in (es.get("validation_users") or []) if "Nymeria_Dataset" in str(v)}
    assert not seen & set(held), "a held-out user was a validation user"
    ds = quiet(SampleDataset, str(NYM), sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
               channels=ck.get("channels", "full"), resample=es.get("resample", "nearest"),
               window_stride=es.get("window_stride"), exclude_users=[str(NYM / u) for u in held], swap_data=True)
    index = SampleIndex(ds, encoding=es.get("encoding", "raw"))
    norm = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if norm.enabled:
        quiet(norm.transform, index)
    assert index.sample_count == EXPECTED_WINDOWS, index.sample_count
    win_user, win_seq, win_script, vocab, names = index_metadata(index, scripts_by_sequence(), held)
    emb = embed(model, index.samples, device)
    res = score_all(emb, win_user, win_seq, win_script, names)
    for proto, p in res.items():
        for c in p["cells"]:
            if proto in ("constrained", "fallback_all48"):
                c["label"] = vocab[c["label"]]
            elif proto == "pair":
                c["label"] = [vocab[x] for x in c["label"]]
            else:
                c["label"] = f"{names[c['label'] // 1000]}/session{c['label'] % 1000}"
    return {"checkpoint": path, "seed": seed, "arm": arm, "device": str(device),
            "gate": {k: v for k, v in g.items() if k != "exclude_users_seen"},
            "users": len(names), "windows": int(index.sample_count), "unseen_datasets": dict(norm.unseen_datasets),
            "encoding": es.get("encoding"), "draws": DRAWS, "draw_seed": DRAW_SEED, "n_small": N_SMALL,
            "scripts": vocab, "protocols": res}


# --------------------------------------------------------------------------------------------------
# Read (committed JSONs only)
# --------------------------------------------------------------------------------------------------

# (lower, upper, meaning) - contiguous over the whole line; scored by WHERE THE INTERVAL FALLS.
REGIONS = {
    "treatment_constrained_n17": [
        (-np.inf, 0.12, "FALSIFIER: the cross-activity person cue does not survive identification"),
        (0.12, 0.20, "BETWEEN: at the k=1 implication of the constrained AUC - gallery averaging across activities buys nothing; report as such"),
        (0.20, 0.45, "BAND: identifies unseen people across activities on glasses, gallery averaging paying as elsewhere"),
        (0.45, np.inf, "ABOVE: exceeding; credit only with the fixture's real-structure activity check passing and the control inside its band")],
    "delta_constrained_n17": [
        (-np.inf, 0.05, "FALSIFIER: in-domain training buys no identification across activities"),
        (0.05, 0.12, "BETWEEN: a real but smaller gain than the verification delta implies; not resolved unless the interval clears 0.05"),
        (0.12, 0.38, "BAND: the verification gain carries to identification"),
        (0.38, np.inf, "ABOVE: exceeding; check the control first - a control below its band produces this for the wrong reason")],
    "control_constrained_n17": [
        (-np.inf, 0.03, "BELOW: anti-identifying with the activity cue neutralised, beyond the fixture's activity envelope - check the harness before reading anything"),
        (0.03, 0.08, "BAND: at chance (1/17 = 0.059) within sampling: no zero-shot person cue across activities (activity alone cannot move this statistic)"),
        (0.08, 0.12, "BETWEEN: a small zero-shot person cue that the adversarial verification pairing (same-script negatives) hid; report beside the delta"),
        (0.12, np.inf, "FALSIFIER: the zero-shot model identifies across activities; the control's sub-chance verification figure was the pairing, not an absent person cue - say so before quoting the delta")],
}


for _name, _rs in REGIONS.items():   # every registered line is partitioned: no unnamed region (CLAUDE.md)
    assert _rs[0][0] == -np.inf and _rs[-1][1] == np.inf, _name
    assert all(a[1] == b[0] for a, b in zip(_rs, _rs[1:])), _name


def regions_hit(name, lo, hi):
    return [m for a, b, m in REGIONS[name] if lo < b and hi >= a]


def verdict(name, r):
    hit, at = regions_hit(name, r["lo"], r["hi"]), regions_hit(name, r["mean"], r["mean"])
    return (f"interval inside one region: {hit[0]}" if len(hit) == 1
            else f"interval spans {len(hit)} regions; the mean sits in: {at[0]}")


def boot_cells(per_user: list[np.ndarray], rng) -> dict:
    """Cluster bootstrap over USERS of the cell-balanced mean: resample users with their cells, then
    mean over the resampled cells. Same N_BOOT and percentiles as the imported user bootstrap."""
    S = np.array([v.sum() for v in per_user])
    C = np.array([len(v) for v in per_user], dtype=float)
    idx = rng.integers(0, len(S), size=(N_BOOT, len(S)))
    stats = S[idx].sum(axis=1) / C[idx].sum(axis=1)
    return {"mean": float(S.sum() / C.sum()), "lo": float(np.percentile(stats, 2.5)), "hi": float(np.percentile(stats, 97.5))}


def _cells(run, proto, setting) -> dict:
    return {u: {int(gi): float(r) for gi, r, _ in v} for u, v in run["protocols"][proto][setting]["cells_per_user"].items()}


def read(paths: list[str]) -> int:
    runs = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    refused = [r for r in runs if "refused" in r]
    for r in refused:
        print(f"REFUSED {r['arm']} s{r['seed']}: {r['refused']} - not read")
    runs = [r for r in runs if "refused" not in r]
    by = {(r["arm"], int(r["seed"])): r for r in runs}
    seeds = sorted({s for a, s in by if a == "treatment"} & {s for a, s in by if a == "control"})
    print(f"paired seeds: {seeds}  (refused: {len(refused)})")
    rng = np.random.default_rng(DRAW_SEED)
    f = lambda d: f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    out = {}
    for proto in PROTOCOLS:
        for setting in ("n17", "full"):
            cells = {arm: [_cells(by[(arm, s)], proto, setting) for s in seeds] for arm in ("treatment", "control")}
            keys = {u: sorted(v) for u, v in cells["treatment"][0].items()}
            for arm in cells:     # the same (user, gallery) cells in every seed and both arms, or nothing is read
                assert all({u: sorted(v) for u, v in c.items()} == keys for c in cells[arm]), f"{proto}/{setting}: cells differ"
            users = sorted(keys)
            if not users:
                print(f"{proto:14s} {setting:4s} no eligible users - nothing to read")
                continue
            vec = {arm: [np.array([np.mean([c[u][g] for c in cells[arm]]) for g in keys[u]]) for u in users] for arm in cells}
            t, c = vec["treatment"], vec["control"]
            run0 = by[("treatment", seeds[0])]["protocols"][proto][setting]
            row = {"users": len(users), "cells": int(sum(len(k) for k in keys.values())), "chance": run0["cell_chance"],
                   # REGISTERED form: cell-balanced
                   "treatment": boot_cells(t, rng), "control": boot_cells(c, rng),
                   "delta": boot_cells([a - b for a, b in zip(t, c)], rng),
                   # secondary: per-user mean (imported user bootstrap) - activity can move this one (fixture 5)
                   "per_user": {arm: ci(np.array([np.mean([by[(arm, s)]["protocols"][proto][setting]["per_user"][u] for s in seeds])
                                                  for u in users]), rng) for arm in ("treatment", "control")},
                   "per_seed": {f"{arm}_s{s}": {k: by[(arm, s)]["protocols"][proto][setting][k] for k in ("cell_mean", "mean", "pooled")}
                                for arm in ("treatment", "control") for s in seeds}}
            if setting == "n17":
                row["implied"] = {f"{arm}_s{s}": [by[(arm, s)]["protocols"][proto]["n17"].get("same_score_auc"),
                                                   by[(arm, s)]["protocols"][proto]["n17"].get("implied_rank1")]
                                  for arm in ("treatment", "control") for s in seeds}
            out[f"{proto}_{setting}"] = row
            print(f"{proto:14s} {setting:4s} users {len(users):2d} cells {row['cells']:3d} chance {row['chance']:.3f} | "
                  f"treatment {f(row['treatment'])} control {f(row['control'])} delta {f(row['delta'])} | "
                  f"per-user t {row['per_user']['treatment']['mean']:.3f} c {row['per_user']['control']['mean']:.3f}")
    reg = {"treatment_constrained_n17": out["constrained_n17"]["treatment"],
           "delta_constrained_n17": out["constrained_n17"]["delta"],
           "control_constrained_n17": out["constrained_n17"]["control"]}
    print("\nregistered quantities, cell-balanced (nymeria_rank1_REGISTERED.md):")
    for k, r in reg.items():
        print(f"  {k}: {f(r)} -> {verdict(k, r)}")
    print("\nimplied vs measured, constrained N=17, same score set (probe-pooled AUC -> implied; probe-pooled measured):")
    for k, (a, i) in out["constrained_n17"]["implied"].items():
        m = out["constrained_n17"]["per_seed"][k]["pooled"]
        print(f"  {k}: AUC {a:.4f} implied {i:.3f} measured {m:.3f} offset {m - i:+.3f}")
    print(f"\nper-user mean is secondary: an activity-only embedding moved it up to {ACTIVITY_FLOOR} from chance on this "
          "script table (fixture 5); the cell-balanced figure cannot be moved by activity alone")
    pathlib.Path(READ_OUT).write_text(json.dumps({"seeds": seeds, "refused": [r["checkpoint"] for r in refused],
                                                  "registered": reg, "rows": out}, indent=1))
    print(f"wrote {READ_OUT}")
    return 0 if not refused else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--checkpoints", nargs="+", required=True, help="SEED=ARM=path (ARM in treatment, control, gnn)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--out-dir", default=str(ROOT / "docs" / "acceptance"))
    r = sub.add_parser("read")
    r.add_argument("paths", nargs="+")
    r.add_argument("--out", default=None, help="where the read summary JSON goes (default docs/acceptance/nymeria_rank1_read.json)")
    a = ap.parse_args()
    if a.cmd == "read":
        global READ_OUT
        READ_OUT = a.out or READ_OUT
        return read(a.paths)
    import torch
    device = torch.device(a.device)
    rc = 0
    for item in a.checkpoints:
        seed, arm, path = item.split("=", 2)
        assert arm in ("treatment", "control", "gnn"), arm     # gnn: nymeria_gnn_REGISTERED.md
        res = score_checkpoint(path, int(seed), arm, device)
        res["registered"] = "docs/acceptance/nymeria_rank1_REGISTERED.md"
        out = pathlib.Path(a.out_dir) / f"nymeria_rank1_{arm}_s{seed}_{device.type}.json"
        out.write_text(json.dumps(res, indent=1))       # written before anything is read from it
        if "refused" in res:
            rc = 1
            print(f"REFUSED {arm} s{seed}: {res['refused']}  -> {out}")
        else:
            c = res["protocols"]["constrained"]
            print(f"{arm:9s} s{seed} gate {res['gate']['gap']:.1e} | constrained N=17 cell-balanced {c['n17']['cell_mean']:.3f} "
                  f"(per-user {c['n17']['mean']:.3f}, {c['n17']['users']} users, {c['n17']['n_cells']} cells)  "
                  f"N=all {c['full']['cell_mean']:.3f} (chance {c['full']['cell_chance']:.3f}) | "
                  f"unconstrained N=17 {res['protocols']['unconstrained']['n17']['mean']:.3f}  -> {out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""
Across-XR: cross-application identification with a train-user-only orthogonal alignment.

Registered first: docs/acceptance/across_xr_alignment_REGISTERED.md. Read that before this.

WHAT THIS SCORES. A dyn checkpoint that never saw Across-XR is embedded on all 49 users x 5
applications. On Schach et al.'s own 17 test users (ids 32-48, from the corpus's `split`
column) it computes rank-1 at N=17 with a gallery template from application A and single
10 s probe windows from application B, for all 20 ordered off-diagonal cells and the 5
within-application cells. Then it fits an orthogonal alignment between application embedding
spaces on users the test users are disjoint from and applies it, unchanged, to the test users.

THE ARMS, all on one set of embeddings so every difference is paired by construction:
  A0        within-application, gallery = first half of the recording, probes = second half
  A1        cross-application, no alignment (PAPER_PLAN P1)
  A2'       alignment fitted on the TEST users themselves - Schach's illegitimate route, the
            diagnostic ceiling on our embedding, never a result
  A2        alignment fitted on users 0-31 (subspace m chosen on 23-31 with fits on 0-22)
  A2-null   the same fit with the user correspondence permuted - a guard that must NOT help
  A2-full   unrestricted 128-d Procrustes on 0-31 - the ill-posed variant, never the headline

WHY THE SUBSPACE. 32 correspondences in 128-d: the cross-covariance has rank <= 32, so the
unrestricted Procrustes R is determined on the fitting centroids' span and is an ARBITRARY
isometry on the 96-d complement - it then rotates test users' embeddings by an accident of
the SVD's null-space basis. A2 therefore solves Procrustes on the top-m principal components
of the fitting users' window embeddings and is the identity on the rest. m is chosen on the
validation split only; the full m-curve is written to the certificate.

TWO GATES before any figure is printed:
  1. checkpoint gate - the checkpoint reproduces its own recorded selected_test_auc on its own
     recorded evaluation users through the pipeline's own loader (honouring test_on_excluded,
     which score_nymeria's loader does not);
  2. fixture gate - tests/unit/test_across_xr_alignment.py runs the same functions on
     synthetic embeddings with a known answer, and asserts on the fixture's content.

    XRSEC_SAMPLE_CACHE_DIR=... python docs/acceptance/across_xr_alignment.py \
        --checkpoints runs/.../best.pth [...] [--device cuda] [--out docs/acceptance/across_xr_alignment.json]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))

XR_DIR = ROOT / "processed_datasets" / "CrossApplicationXR_Dataset"
XR_USERS = XR_DIR / "users"
GAMES = ("superhot_vr", "half_life_alyx", "beat_saber", "synth_riders", "social_vr")
# Our training activities on other corpora (BOXRR is Beat Saber, who_is_alyx is Alyx).
SEEN_ACTIVITIES = {"beat_saber", "half_life_alyx"}
M_GRID = (4, 8, 16, 24, 32)
N_BOOT = 10000
SEQUENCE_SECONDS = 600.0
GATE_TOLERANCE = {"cuda": 1e-4, "cpu": 2e-3}


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# --------------------------------------------------------------------------------------
# Pure numpy: alignment and scoring. These are what the fixture tests exercise.
# --------------------------------------------------------------------------------------

def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def centroids(emb: np.ndarray, rows_per_user: list[np.ndarray]) -> np.ndarray:
    """One renormalised mean of normalised embeddings per user; rows_per_user gives each
    user's window rows. Users with no rows raise rather than produce a zero row."""
    out = []
    for rows in rows_per_user:
        if len(rows) == 0:
            raise ValueError("a user has no windows in this application")
        out.append(l2(l2(emb[rows]).mean(axis=0)))
    return np.stack(out)


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||source @ R - target||_F (rotation or reflection).
    R = U V^T where U S V^T = source^T target."""
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=True)
    return u @ vt


def pca_basis(x: np.ndarray, m: int) -> np.ndarray:
    """d x m orthonormal basis of the top-m principal components of the rows of x."""
    centred = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:m].T


def subspace_procrustes(source: np.ndarray, target: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Procrustes solved inside span(basis) and the identity on its complement. Returns the
    full d x d orthogonal matrix, so callers apply it exactly as they would the unrestricted one."""
    d = basis.shape[0]
    r_m = procrustes(source @ basis, target @ basis)
    return basis @ r_m @ basis.T + (np.eye(d) - basis @ basis.T)


def rank1_per_user(gallery: np.ndarray, probes: np.ndarray, probe_user: np.ndarray,
                   n_users: int) -> np.ndarray:
    """Per-user rank-1 accuracy (n_users,). Cosine on L2-normalised inputs; ties are
    rank-averaged, as everywhere in this repo, so a constant scorer is not rank-1."""
    scores = l2(probes) @ l2(gallery).T                     # (probes, users)
    own = scores[np.arange(len(probes)), probe_user]
    better = (scores > own[:, None]).sum(axis=1)
    ties = (scores == own[:, None]).sum(axis=1) - 1
    rank = 1 + better + 0.5 * ties
    hit = rank <= 1.0
    acc = np.zeros(n_users)
    for u in range(n_users):
        mask = probe_user == u
        acc[u] = hit[mask].mean() if mask.any() else np.nan
    return acc


def rank1_predictions(gallery: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Nearest-template index per probe, for the majority-vote sequence metric."""
    return np.argmax(l2(probes) @ l2(gallery).T, axis=1)


def bootstrap_mean(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Cluster bootstrap over the leading axis (users). Returns (mean, lo95, hi95)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------------------
# Corpus bookkeeping
# --------------------------------------------------------------------------------------

class Corpus:
    """All 49 users of Across-XR at the checkpoint's own resolution and encoding, with every
    window labelled by user id, application and start time."""

    def __init__(self, ck: dict, device, model, batch: int = 512):
        from dataset import SampleDataset, SampleIndex
        from normalization import ChannelNormalizer

        es = ck["eval_split"]
        self.sample_time = int(es["sample_time"])
        self.sample_rate = int(es["sample_rate"])
        self.stride = es.get("window_stride")
        self.encoding = es.get("encoding", "raw")
        ds = quiet(SampleDataset, str(XR_USERS), sample_time=self.sample_time,
                   sample_rate=self.sample_rate, channels=ck.get("channels", "full"),
                   resample=es.get("resample", "nearest"), window_stride=self.stride)
        index = SampleIndex(ds, encoding=self.encoding)
        normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
        if normalizer.enabled:
            quiet(normalizer.transform, index)
        self.unseen_policy = dict(normalizer.unseen_datasets)

        # user id from the directory name; application from the sorted CSV list, which is
        # what session_ids enumerate (only loaded files - asserted to be all five).
        self.user_ids = np.array([int(pathlib.Path(d).name) for d in ds.user_dirs])
        assert len(self.user_ids) == 49, f"expected 49 users, loaded {len(self.user_ids)}"
        assert len(set(self.user_ids.tolist())) == 49
        n_windows = index.sample_count
        self.window_user = np.empty(n_windows, dtype=int)     # position in self.user_ids
        self.window_app = np.empty(n_windows, dtype=object)
        self.window_start = index.window_start_times.numpy().astype(float)
        sessions = index.window_session_ids.numpy()
        for u, (user_dir, rows) in enumerate(zip(ds.user_dirs, index.user_sample_indices)):
            csvs = sorted(f for f in os.listdir(user_dir) if f.endswith(".csv"))
            apps = [next(g for g in GAMES if f.startswith(g)) for f in csvs]
            assert len(apps) == 5 and set(apps) == set(GAMES), (user_dir, csvs)
            rows = rows.numpy()
            sid = sessions[rows]
            assert set(sid.tolist()) == set(range(5)), (user_dir, sorted(set(sid.tolist())))
            self.window_user[rows] = u
            self.window_app[rows] = np.array(apps, dtype=object)[sid]

        splits = json.loads((XR_DIR / "splits.json").read_text(encoding="utf-8"))
        self.split = {name: sorted(int(i) for i in splits[name]) for name in ("train", "valid", "test")}
        assert self.split["test"] == list(range(32, 49)), self.split["test"]
        assert self.split["train"] == list(range(0, 23)) and self.split["valid"] == list(range(23, 32))

        self.embeddings = embed(model, index.samples, device, batch)
        self.raw_positions = None

    def rows(self, user_ids, app: str, half: str | None = None) -> list[np.ndarray]:
        """Window rows per user (in the given order) for one application; half selects the
        first or second half of each user's recording by start time."""
        out = []
        for uid in user_ids:
            u = int(np.where(self.user_ids == uid)[0][0])
            mask = (self.window_user == u) & (self.window_app == app)
            r = np.where(mask)[0]
            if half is not None:
                cut = np.median(self.window_start[r])
                r = r[self.window_start[r] < cut] if half == "first" else r[self.window_start[r] >= cut]
            out.append(r)
        return out


def embed(model, samples: torch.Tensor, device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            out.append(model.feature_extractor(samples[i:i + batch].to(device)).float().cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------------------
# One cell = (gallery application A, probe application B, alignment R or None)
# --------------------------------------------------------------------------------------

def score_cell(corpus: Corpus, users: list[int], app_a: str, app_b: str,
               r: np.ndarray | None) -> dict:
    """Per-user rank-1 for gallery from A and probes from B over `users` (N = len(users)),
    plus the 10-minute majority-vote decision per user. Within-application cells split the
    recording in half by start time."""
    emb = corpus.embeddings
    if app_a == app_b:
        g_rows = corpus.rows(users, app_a, half="first")
        p_rows = corpus.rows(users, app_b, half="second")
    else:
        g_rows = corpus.rows(users, app_a)
        p_rows = corpus.rows(users, app_b)
    gallery = centroids(emb, g_rows)
    probe_rows = np.concatenate(p_rows)
    probe_user = np.concatenate([np.full(len(r), i) for i, r in enumerate(p_rows)])
    probes = emb[probe_rows]
    if r is not None:
        probes = probes @ r
    per_user = rank1_per_user(gallery, probes, probe_user, len(users))
    # 10-minute sequence: majority vote over the first SEQUENCE_SECONDS of each user's probe
    # stream (one decision per user per cell - coarse by construction, see the registration).
    pred = rank1_predictions(gallery, probes)
    starts = corpus.window_start[probe_rows]
    seq_hit = np.zeros(len(users))
    for i in range(len(users)):
        mask = (probe_user == i)
        t0 = starts[mask].min()
        window = mask & (starts < t0 + SEQUENCE_SECONDS)
        votes = np.bincount(pred[window], minlength=len(users))
        top = np.flatnonzero(votes == votes.max())
        seq_hit[i] = 1.0 if (len(top) == 1 and top[0] == i) else (1.0 / len(top) if i in top else 0.0)
    return {"per_user": per_user, "sequence": seq_hit, "probes": int(len(probe_rows))}


def fit_alignment(corpus: Corpus, fit_users: list[int], app_a: str, app_b: str,
                  m: int | None, permute: np.random.Generator | None = None) -> np.ndarray:
    """R mapping application-B embeddings into application-A space, fitted on fit_users'
    centroids. m=None is the unrestricted fit; otherwise Procrustes on the top-m PCA
    components of the fitting users' window embeddings (A and B pooled). `permute` shuffles
    the B centroids' user correspondence - the null arm."""
    emb = corpus.embeddings
    a_rows, b_rows = corpus.rows(fit_users, app_a), corpus.rows(fit_users, app_b)
    c_a, c_b = centroids(emb, a_rows), centroids(emb, b_rows)
    if permute is not None:
        c_b = c_b[permute.permutation(len(c_b))]
    if m is None:
        return procrustes(c_b, c_a)
    pooled = l2(emb[np.concatenate(a_rows + b_rows)])
    return subspace_procrustes(c_b, c_a, pca_basis(pooled, m))


def cross_cells():
    return [(a, b) for a in GAMES for b in GAMES if a != b]


def mean_over_cells(results: dict[tuple, dict], key: str = "per_user") -> np.ndarray:
    """Per-user vector averaged over the cells given: the unit the bootstrap resamples."""
    return np.mean([results[c][key] for c in results], axis=0)


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------

def gate(ckpt_path: str, device) -> dict:
    """Reproduce the checkpoint's recorded selected_test_auc on its recorded evaluation users
    through the pipeline's own loader. Honours test_on_excluded / swap_data exactly as
    create_dataloader_from_path's test branch does."""
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import results_log
    from dataset import SiameseDataset, _seed_value
    from eval import evaluate
    from normalization import ChannelNormalizer
    from utils import load_checkpoint

    shard = ROOT / "results" / "runs" / f"{results_log.machine_name()}.jsonl"
    rel = str(pathlib.Path(ckpt_path).resolve().relative_to(ROOT)).replace("\\", "/")
    rows = [json.loads(l) for l in shard.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if (r.get("checkpoint") or "").replace("\\", "/") == rel and r.get("mode") == "train"]
    if not rows:
        return {"checkpoint": rel, "passed": False, "reason": "no training row in the shard"}
    row = rows[-1]
    model, ck = quiet(load_checkpoint, ckpt_path, device, 100, return_checkpoint=True)
    es = ck["eval_split"]
    seed = int(ck.get("seed", row["seed"]))
    swap = (not bool(es.get("swap_data", False))) if bool(es.get("test_on_excluded", False)) else bool(es.get("swap_data", False))
    dataset = quiet(SiameseDataset, list(es["test_dirs"]), samples_per_user=int(row.get("samples_per_user") or 512),
                    sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
                    exclude_users=list(es.get("exclude_users") or []), swap_data=swap,
                    seed=_seed_value(seed, 4), within_dataset_negatives=bool(row.get("within_dataset_negatives", True)),
                    channels=ck.get("channels", "full"), resample=es.get("resample", "nearest"),
                    window_stride=es.get("window_stride"), cross_session_positives=bool(row.get("cross_session_positives", True)),
                    encoding=es.get("encoding", "raw"))
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen=row.get("eval_normalize") or "target_fit")
    if normalizer.enabled:
        quiet(normalizer.transform, dataset.sample_index)
    dataset.unseen_datasets = dict(normalizer.unseen_datasets)
    loader = DataLoader(dataset, batch_size=int(row.get("batch_size") or 256), shuffle=False)
    _, _, metrics = quiet(evaluate, model, loader, nn.BCEWithLogitsLoss(), device, return_metrics=True)
    gap = abs(float(metrics["auc"]) - float(row["selected_test_auc"]))
    tol = GATE_TOLERANCE[device.type]
    result = {"checkpoint": rel, "run_id": row["run_id"], "seed": seed, "device": str(device),
              "recorded": row["selected_test_auc"], "rescored": metrics["auc"], "gap": gap,
              "tolerance": tol, "eval_users": dataset.num_users,
              "position_lookup_recorded": row.get("position_lookup_auc"),
              "position_lookup_rescored": metrics.get("position_lookup_auc"), "passed": gap <= tol}
    print(f"gate {rel}: recorded {row['selected_test_auc']:.6f} rescored {metrics['auc']:.6f} "
          f"on {device} ({dataset.num_users} users) gap {gap:.1e} {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    return result, model, ck


# --------------------------------------------------------------------------------------
# Main: one checkpoint = one seed
# --------------------------------------------------------------------------------------

def run_seed(ckpt_path: str, device, rng: np.random.Generator, skip_gate: bool = False) -> dict:
    from utils import load_checkpoint
    if skip_gate:
        model, ck = quiet(load_checkpoint, ckpt_path, device, 100, return_checkpoint=True)
        g = {"checkpoint": ckpt_path, "passed": None, "reason": "gate skipped by flag"}
    else:
        g, model, ck = gate(ckpt_path, device)
        if not g["passed"]:
            return {"gate": g}
        # test_dirs plus test_on_excluded=true keeps ONLY the excluded users under test_dirs;
        # an exclude path that points anywhere else loads 0 users and the only tell is a
        # stdout line Hydra does not capture. Exactly Schach's 17, or nothing is read.
        assert g["eval_users"] == 17, f"the gate loader saw {g['eval_users']} users, not 17"
    t0 = time.time()
    corpus = Corpus(ck, device, model)
    print(f"  embedded {len(corpus.embeddings)} windows ({corpus.encoding}, {corpus.sample_time}s, "
          f"stride {corpus.stride}) in {time.time() - t0:.0f}s", flush=True)
    test, train, valid = corpus.split["test"], corpus.split["train"], corpus.split["valid"]
    fit_all = train + valid

    out = {"gate": g, "seed": int(ck.get("seed", 0)), "encoding": corpus.encoding,
           "windows": int(len(corpus.embeddings)), "unseen_policy": corpus.unseen_policy}

    # A0 and A1 need no fit.
    a0 = {(a, a): score_cell(corpus, test, a, a, None) for a in GAMES}
    a1 = {c: score_cell(corpus, test, c[0], c[1], None) for c in cross_cells()}

    # m chosen on the validation split: fit on train, score on valid (N=9).
    m_curve = {}
    for m in M_GRID:
        cells = {}
        for a, b in cross_cells():
            r = fit_alignment(corpus, train, a, b, m)
            cells[(a, b)] = score_cell(corpus, valid, a, b, r)
        m_curve[m] = float(np.nanmean(mean_over_cells(cells)))
    valid_unaligned = float(np.nanmean(mean_over_cells(
        {c: score_cell(corpus, valid, c[0], c[1], None) for c in cross_cells()})))
    m_star = max(M_GRID, key=lambda m: m_curve[m])
    # Nine validation users: rank-1 there has a per-user sd near 0.15, and the argmax
    # over five m values is a max over noisy draws. Whether the curve is FLAT is the
    # informative fact; a peak chosen on nine people is not a tuned value.
    m_range = max(m_curve.values()) - min(m_curve.values())
    flat = m_range < 0.05
    print(f"  m-curve on validation (N=9): unaligned {valid_unaligned:.3f}, "
          + ", ".join(f"m={m}: {v:.3f}" for m, v in m_curve.items())
          + f" -> m*={m_star}; range {m_range:.3f} ({'FLAT - the choice is immaterial' if flat else 'PEAKED - m is unresolved on 9 users'})",
          flush=True)
    out["m_curve_valid_n9"] = {"unaligned": valid_unaligned, **{str(m): v for m, v in m_curve.items()},
                               "m_star": m_star, "range": m_range, "flat_below_0.05": flat}

    arms = {"A0": a0, "A1": a1}
    arms["A2"] = {c: score_cell(corpus, test, c[0], c[1], fit_alignment(corpus, fit_all, c[0], c[1], m_star))
                  for c in cross_cells()}
    arms["A2prime"] = {c: score_cell(corpus, test, c[0], c[1], fit_alignment(corpus, test, c[0], c[1], m_star))
                       for c in cross_cells()}
    null_rng = np.random.default_rng(int(ck.get("seed", 0)) + 1000)
    arms["A2null"] = {c: score_cell(corpus, test, c[0], c[1], fit_alignment(corpus, fit_all, c[0], c[1], m_star, permute=null_rng))
                      for c in cross_cells()}
    arms["A2full"] = {c: score_cell(corpus, test, c[0], c[1], fit_alignment(corpus, fit_all, c[0], c[1], None))
                      for c in cross_cells()}

    summary = {}
    for name, cells in arms.items():
        vec = mean_over_cells(cells)
        mean, lo, hi = bootstrap_mean(vec, N_BOOT, rng)
        seq = float(np.mean([cells[c]["sequence"].mean() for c in cells]))
        per_cell = {f"{a}->{b}": float(np.nanmean(cells[(a, b)]["per_user"])) for (a, b) in cells}
        unseen = [c for c in cells if c[0] not in SEEN_ACTIVITIES and c[1] not in SEEN_ACTIVITIES and c[0] != c[1]]
        seen = [c for c in cells if c not in unseen and c[0] != c[1]]
        summary[name] = {
            "mean": mean, "ci95": [lo, hi], "sequence_10min": seq, "per_cell": per_cell,
            "per_user": vec.tolist(),
            "unseen_activity_cells": float(np.nanmean(mean_over_cells({c: cells[c] for c in unseen}))) if unseen else None,
            "seen_activity_cells": float(np.nanmean(mean_over_cells({c: cells[c] for c in seen}))) if seen else None,
        }
        print(f"  {name:8s} rank-1@17 {mean:.3f} [{lo:.3f}, {hi:.3f}]  10-min {seq:.3f}", flush=True)

    # Paired differences, bootstrapped over users on the same 20 cells.
    def paired(x, y):
        d = np.array(summary[x]["per_user"]) - np.array(summary[y]["per_user"])
        return dict(zip(("mean", "lo", "hi"), bootstrap_mean(d, N_BOOT, rng)))
    out["paired"] = {
        "A2prime-A1": paired("A2prime", "A1"), "A2-A1": paired("A2", "A1"),
        "A2-A2prime": paired("A2", "A2prime"), "A2null-A1": paired("A2null", "A1"),
        "A2full-A2": paired("A2full", "A2"),
    }
    for k, v in out["paired"].items():
        print(f"  {k:12s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]", flush=True)
    out["arms"] = summary
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "docs" / "acceptance" / "across_xr_alignment.json"))
    ap.add_argument("--skip-gate", action="store_true", help="development only; the certificate says so")
    args = ap.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(67)
    results = []
    for ck in args.checkpoints:
        print(f"\n=== {ck} ===", flush=True)
        results.append(run_seed(ck, device, rng, skip_gate=args.skip_gate))
    gates = [r["gate"] for r in results]
    gate_path = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + "_gate.json")
    gate_path.write_text(json.dumps(gates, indent=1), encoding="utf-8")
    pathlib.Path(args.out).write_text(json.dumps(
        {"registered": "docs/acceptance/across_xr_alignment_REGISTERED.md", "device": str(device),
         "seeds": results}, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {args.out} and {gate_path}")
    return 0 if all(r.get("gate", {}).get("passed") is not False for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

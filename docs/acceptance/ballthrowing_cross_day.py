"""
Ball-throwing: identification across days and across headsets (ballthrowing_cross_day_REGISTERED.md).

Per checkpoint: gate it on its own evaluation users (the alignment harness's gate), embed every throw of
the 41 participants through the pipeline's own SampleDataset / SampleIndex / normaliser at the checkpoint's
settings (one 2 s window per throw), and score rank-1 at N=41 with the A1 rule (`centroids`,
`rank1_per_user` imported from across_xr_alignment) under three conditions:

  C0 same session             gallery throws 0-4 / probe throws 5-9 of one session (and the reverse), 6 sessions
  C1 same headset, other day  quest1<->quest2, vive1<->vive2, cosmos1<->cosmos2, both directions
  C2 other headset, other day the 12 cross-headset session pairs, both directions
  C2<=3d (descriptive)        C2 restricted per user to pairs at most 3 days apart (day_gaps.csv)

plus the training-free height-only lookup (descriptive) in C1 and C2: per-window mean recorded head y,
standardised per session type across the 41 people (removes each headset's scene-origin offset).

    python docs/acceptance/ballthrowing_cross_day.py --checkpoints SEED=path [...] --device cuda --out ...
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
from across_xr_alignment import centroids, embed, gate, quiet, rank1_per_user  # noqa: E402

CORPUS = ROOT / "processed_datasets" / "BallThrowing"
SESSIONS = ("quest1", "quest2", "vive1", "vive2", "cosmos1", "cosmos2")
_abbr = {"quest": "Q", "vive": "V", "cosmos": "C"}
GATE_TOL = {"cpu": 1e-3, "cuda": 1e-4}


def gap_column(a, b):
    """day_gaps.csv column for a session pair, in the file's own Q < V < C, day order."""
    order = {s: i for i, s in enumerate(SESSIONS)}
    a, b = sorted((a, b), key=order.get)
    return f"{_abbr[a[:-1]]}{a[-1]}{_abbr[b[:-1]]}{b[-1]}Diff"


C1_PAIRS = [("quest1", "quest2"), ("vive1", "vive2"), ("cosmos1", "cosmos2")]
C2_PAIRS = [(a, b) for a, b in itertools.combinations(SESSIONS, 2) if a[:-1] != b[:-1]]
assert len(C2_PAIRS) == 12


def load_corpus(ck, model, device):
    from dataset import SampleDataset, SampleIndex
    from normalization import ChannelNormalizer
    es = ck["eval_split"]
    ds = quiet(SampleDataset, str(CORPUS / "users"), sample_time=int(es["sample_time"]),
               sample_rate=int(es["sample_rate"]), channels=ck.get("channels", "full"),
               resample=es.get("resample", "nearest"), window_stride=es.get("window_stride"))
    index = SampleIndex(ds, encoding=es.get("encoding", "raw"))
    assert len(ds.user_dirs) == 41, len(ds.user_dirs)
    assert index.sample_count == 41 * 60, f"expected one window per throw (2,460), got {index.sample_count}"
    heights = index.window_mean_positions[:, 1].numpy().astype(float)        # recorded, before encoding
    norm = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if norm.enabled:
        quiet(norm.transform, index)
    sess_ids = index.window_session_ids.numpy()
    users, where = [], {}
    for u, (user_dir, rows) in enumerate(zip(ds.user_dirs, index.user_sample_indices)):
        name = pathlib.Path(user_dir).name
        users.append(name)
        csvs = sorted(f for f in os.listdir(user_dir) if f.endswith(".csv"))
        for r in rows.numpy():
            stem = csvs[sess_ids[r]][:-4]
            session, throw = stem.split("_throw")
            where[(u, session, int(throw))] = int(r)
    assert len(where) == 2460
    return embed(model, index.samples, device), heights, users, where, dict(norm.unseen_datasets)


def rows(where, u, session, throws=range(10)):
    return np.array([where[(u, session, t)] for t in throws])


def rank1(emb, gal_rows, probe_rows):
    n = len(gal_rows)
    gallery = centroids(emb, gal_rows)
    probe_user = np.concatenate([np.full(len(p), i) for i, p in enumerate(probe_rows)])
    return rank1_per_user(gallery, emb[np.concatenate(probe_rows)], probe_user, n)


def height_rank1(h_std, gal_rows, probe_rows):
    """rank-1 of |probe window height - gallery session mean height|, ties rank-averaged."""
    gal = np.array([h_std[r].mean() for r in gal_rows])
    out = np.zeros(len(gal_rows))
    for i, pr in enumerate(probe_rows):
        d = np.abs(h_std[pr][:, None] - gal[None, :])            # (probes, users)
        own = d[:, i]
        better = (d < own[:, None]).sum(axis=1); ties = (d == own[:, None]).sum(axis=1) - 1
        out[i] = float(np.mean((better == 0) & (ties == 0)))     # the rank1_per_user rule: rank <= 1
    return out


def score(emb, heights, users, where):
    n = len(users)
    U = range(n)
    gaps = {}
    with open(CORPUS / "day_gaps.csv") as f:
        for row in csv.DictReader(f):
            gaps[row["ID"].strip()] = row
    # per-session-type standardisation of recorded head height
    h_std = heights.copy()
    for s in SESSIONS:
        idx = np.concatenate([rows(where, u, s) for u in U])
        h_std[idx] = (heights[idx] - heights[idx].mean()) / heights[idx].std()
    res = {}
    c0 = []
    for s in SESSIONS:
        a = [rows(where, u, s, range(0, 5)) for u in U]; b = [rows(where, u, s, range(5, 10)) for u in U]
        c0.append((rank1(emb, a, b) + rank1(emb, b, a)) / 2)
    res["C0"] = np.mean(c0, axis=0)
    per_pair = {}
    for a, b in C1_PAIRS + C2_PAIRS:
        ga = [rows(where, u, a) for u in U]; gb = [rows(where, u, b) for u in U]
        per_pair[(a, b)] = {"model": (rank1(emb, ga, gb) + rank1(emb, gb, ga)) / 2,
                            "height": (height_rank1(h_std, ga, gb) + height_rank1(h_std, gb, ga)) / 2}
    for cond, pairs in (("C1", C1_PAIRS), ("C2", C2_PAIRS)):
        res[cond] = np.mean([per_pair[p]["model"] for p in pairs], axis=0)
        res[f"{cond}_height"] = np.mean([per_pair[p]["height"] for p in pairs], axis=0)
    # gap-matched C2: per user, only cross-headset pairs at most 3 days apart
    matched = np.full(n, np.nan)
    for i, u in enumerate(users):
        vals = [per_pair[p]["model"][i] for p in C2_PAIRS if int(gaps[u][gap_column(*p)]) <= 3]
        if vals:
            matched[i] = float(np.mean(vals))
    res["C2_le3d"] = matched
    out = {k: dict(zip(users, map(float, v))) for k, v in res.items()}
    out["per_pair_model_mean"] = {f"{a}-{b}": float(v["model"].mean()) for (a, b), v in per_pair.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="SEED=path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device)
    results = []
    for item in a.checkpoints:
        seed, path = item.split("=", 1)
        g, model, ck = gate(path, device)
        tol = GATE_TOL[device.type]
        if not g.get("passed") or g.get("gap") is None or g["gap"] > tol:
            results.append({"seed": int(seed), "checkpoint": path, "refused": f"gate gap {g.get('gap')} > {tol} ({g.get('reason', '')})"})
            print("REFUSED", seed, results[-1]["refused"])
        else:
            emb, heights, users, where, unseen = load_corpus(ck, model, device)
            r = score(emb, heights, users, where)
            results.append({"seed": int(seed), "checkpoint": path, "gate": {k: v for k, v in g.items() if k != "exclude_users_seen"},
                            "unseen_policy": unseen, "n_users": len(users), "chance": 1 / len(users), **r})
            m = {k: np.nanmean(list(r[k].values())) for k in ("C0", "C1", "C2", "C2_le3d", "C1_height", "C2_height")}
            print(f"seed {seed} gate {g['gap']:.1e} | " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
        pathlib.Path(a.out).write_text(json.dumps({"registered": "docs/acceptance/ballthrowing_cross_day_REGISTERED.md",
                                                   "device": str(device), "results": results}, indent=1))
    print(f"wrote {a.out}")
    return 0 if all("refused" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

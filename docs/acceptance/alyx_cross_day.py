"""
Cross-day identification on who_is_alyx (alyx_cross_day_REGISTERED.md).

For each checkpoint: gate it on its own evaluation users (the alignment harness's gate), then embed
its own alyx validation users who have two sessions on two different dates, through the pipeline's own
SampleDataset / SampleIndex / the checkpoint's normaliser, and score rank-1 at N = every eligible user:

  cross-day  gallery = one day's session, probe = the other day's windows, both directions averaged
  same-day   gallery = first half of a session, probe = its second half, both sessions averaged

A1 rule throughout (centroid template of L2-normalised window embeddings, cosine, ties rank-averaged;
`centroids` / `rank1_per_user` imported from across_xr_alignment). A same-day split never lets a window
straddle the cut: first-half windows must END by the median start, so no probe shares frames with a
gallery window (the pipeline's own self-match guard, applied here).

    python docs/acceptance/alyx_cross_day.py --checkpoints SEED=ARM=path [...] --device cpu --out ...
"""
from __future__ import annotations

import argparse
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
from exposure_breadth_read import ci  # noqa: E402

ALYX = ROOT / "processed_datasets" / "who_is_alyx" / "users"
GATE_TOL = {"cpu": 1e-3, "cuda": 1e-4}


def eligible_users(seed: int) -> list[str]:
    ref = json.loads((ROOT / "docs/acceptance" / f"nymeria_in_domain_lists_s{seed}.json").read_text())
    out = []
    for item in ref["validation_control"]:
        corpus, u = item.split("/", 1)
        if corpus != "who_is_alyx":
            continue
        csvs = sorted(f for f in os.listdir(ALYX / u) if f.endswith(".csv"))
        if len(csvs) == 2:
            assert csvs[0] != csvs[1] and csvs[0][:10] != csvs[1][:10], (u, csvs)   # two different dates
            out.append(str(ALYX / u))
    return sorted(out)


def score_checkpoint(path: str, seed: int, device) -> dict:
    from dataset import SampleDataset, SampleIndex
    from normalization import ChannelNormalizer
    g, model, ck = gate(path, device)
    tol = GATE_TOL[device.type]
    if not g.get("passed") or g.get("gap") is None or g["gap"] > tol:
        return {"checkpoint": path, "refused": f"gate gap {g.get('gap')} > {tol} on {device} ({g.get('reason', '')})"}
    es = ck["eval_split"]
    users = eligible_users(seed)
    # never trained on: not in the checkpoint's training, i.e. listed as validation (or excluded) there
    val = set(es.get("validation_users") or [])
    if val:
        names = {pathlib.Path(v).name for v in val if "who_is_alyx" in v}
        assert all(pathlib.Path(u).name in names for u in users), "an eligible alyx user is not a validation user of this checkpoint"
    ds = quiet(SampleDataset, str(ALYX), sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
               channels=ck.get("channels", "full"), resample=es.get("resample", "nearest"),
               window_stride=es.get("window_stride"), exclude_users=users, swap_data=True)
    assert sorted(ds.user_dirs) == users, (len(ds.user_dirs), len(users))
    index = SampleIndex(ds, encoding=es.get("encoding", "raw"))
    norm = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if norm.enabled:
        quiet(norm.transform, index)
    assert not norm.unseen_datasets, f"who_is_alyx target-fitted: {norm.unseen_datasets}"
    emb = embed(model, index.samples, device)
    sess = index.window_session_ids.numpy()
    start = index.window_start_times.numpy().astype(float)
    T = float(es["sample_time"])
    rows = []
    for r in index.user_sample_indices:
        r = r.numpy()
        s0, s1 = r[sess[r] == 0], r[sess[r] == 1]
        assert len(s0) and len(s1)
        halves = []
        for s in (s0, s1):
            cut = np.median(start[s])
            first, second = s[start[s] + T <= cut], s[start[s] >= cut]
            assert len(first) and len(second)
            halves.append((first, second))
        rows.append({"s0": s0, "s1": s1, "halves": halves})
    n = len(rows)

    def rank1(gal, prb):
        gallery = centroids(emb, gal)
        probe_user = np.concatenate([np.full(len(p), i) for i, p in enumerate(prb)])
        return rank1_per_user(gallery, emb[np.concatenate(prb)], probe_user, n)

    cross = (rank1([x["s0"] for x in rows], [x["s1"] for x in rows]) +
             rank1([x["s1"] for x in rows], [x["s0"] for x in rows])) / 2
    same = (rank1([x["halves"][0][0] for x in rows], [x["halves"][0][1] for x in rows]) +
            rank1([x["halves"][1][0] for x in rows], [x["halves"][1][1] for x in rows])) / 2
    names = [pathlib.Path(u).name for u in ds.user_dirs]
    return {"checkpoint": path, "seed": seed, "gate": {k: v for k, v in g.items() if k != "exclude_users_seen"},
            "n_users": n, "chance": 1.0 / n, "windows": int(len(emb)),
            "cross_day": dict(zip(names, map(float, cross))), "same_day": dict(zip(names, map(float, same)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="SEED=ARM=path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device)
    results = []
    for item in a.checkpoints:
        seed, arm, path = item.split("=", 2)
        r = score_checkpoint(path, int(seed), device)
        r["arm"] = arm
        results.append(r)
        if "refused" in r:
            print(f"REFUSED {arm} s{seed}: {r['refused']}")
        else:
            c, s = np.mean(list(r["cross_day"].values())), np.mean(list(r["same_day"].values()))
            print(f"{arm:22s} s{seed} N={r['n_users']} gate {r['gate']['gap']:.1e} | cross-day {c:.3f}  same-day {s:.3f}  cost {c - s:+.3f}")
        pathlib.Path(a.out).write_text(json.dumps({"registered": "docs/acceptance/alyx_cross_day_REGISTERED.md",
                                                   "device": str(device), "results": results}, indent=1))
    print(f"wrote {a.out}")
    return 0 if all("refused" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

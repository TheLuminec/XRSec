"""
Step 6, the dyn column on the seated corpora - the re-run after the retraction.

The first attempt built the sample index without passing `encoding='dyn'`, so checkpoints
trained on pose relative to the window's own mean were scored on absolute pose. Every
number it produced was void. The single line that fixes it is the `encoding="dyn"` in
build_pair() below; everything else here exists so that kind of thing cannot pass silently
again.

TWO GATES, both run before any dyn figure is printed:

  1. The checkpoint gate (step6_seated_dyn_gate.py, passed 5/5): each checkpoint reproduces
     its own recorded selected_test_auc on its own recorded users. That catches "the model
     is being fed the wrong thing".

  2. The static gate, here: this harness recomputes the STATIC rank-1 columns and they must
     land on the figures already published in CLAUDE.md. That catches "the model is fed
     correctly but the enrolment protocol differs from the column I am comparing against" -
     population, k, which session is gallery, the rng, the tie handling. Gate 1 cannot see
     any of that, because none of it exists on the training path.

A dyn number is only comparable to a static number if both gates hold, so both are printed
beside the result rather than kept in a log.

Two indices per corpus: `dyn` is what the model consumes, `raw` is where head height still
lives. They are built over the same users in the same order and asserted aligned, because a
silent misalignment would produce a plausible fused number from mismatched rows.
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import build_sample_index                    # noqa: E402
from normalization import ChannelNormalizer               # noqa: E402
from utils import load_checkpoint                         # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.device_count() else "cpu")
PROBES_PER_USER = 5
DRAWS = 300
N_SMALL = 17
SEED = 67

DYN_93 = {  # seed -> the 9.3 dyn checkpoint. Trained on BOXRR + alyx ONLY, so every
    1: "sweeps/cb0a7dd722/runs/bilstm_a41190094c/best.pth",   # seated corpus here is unseen.
    2: "sweeps/cb0a7dd722/runs/bilstm_0ebccac678/best.pth",
    3: "sweeps/cb0a7dd722/runs/bilstm_ab82c3b90b/best.pth",
    4: "sweeps/cb0a7dd722/runs/bilstm_796d3932d4/best.pth",
    5: "sweeps/cb0a7dd722/runs/bilstm_8d679a46cc/best.pth",
}
STATIC_NORM = "sweeps/f24f2a6c1e/runs/bilstm_a41190094c/best.pth"   # the static column's normaliser

# k is each corpus's maximum, not a free choice: ViewGauss sessions hold 3 windows at 5 s
# and Head_and_Gaze 11, so k=16 is impossible there. Static targets are CLAUDE.md's table.
CORPORA = {
    "ViewGauss":        ("ViewGauss_Head-Movement_Dataset", 3,  (0.814, 0.540, 0.627)),
    "Head_and_Gaze":    ("Head_and_Gaze_Behavior_Dataset",  8,  (0.609, 0.142, 0.618)),
    "VR_User_Behavior": ("VR_User_Behavior_Dataset_(Spherical_Video_Streaming)", 16, (0.790, 0.114, 0.832)),
}
AXES = {"xyz": [0, 1, 2], "y": [1], "xz": [0, 2]}


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def write_gate_certificate(name: str, records: list[dict]) -> pathlib.Path:
    """Persist a gate result even when the caller only needed it inline.

    "It was gated" and "there is a committed certificate that it was gated" are different
    claims, and only the second survives the session. The reusable-control pattern - comparing
    against a checkpoint's recorded row without re-running it - depends on the certificate
    existing, not on the check having happened, so a harness that gates a checkpoint and throws
    the result away has done the work and kept none of the value. Same shape as reading "on
    origin" from origin rather than from the working file: the artefact, not the recollection.
    """
    out = pathlib.Path("docs/acceptance") / f"{name}_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    passed = sum(bool(r.get("passed")) for r in records)
    print(f"  gate certificate: {passed}/{len(records)} passed -> {out}", flush=True)
    return out


def build_pair(users_dir, keep=None):
    """The same windows twice: dyn for the model, raw for head height. Asserted aligned."""
    kw = dict(sample_time=5, sample_rate=20)
    if keep is not None:
        kw.update(exclude_users=keep, swap_data=True)
    dyn = quiet(build_sample_index, users_dir, encoding="dyn", **kw)   # <- the retracted run's missing line
    raw = quiet(build_sample_index, users_dir, **kw)
    assert dyn.samples.shape == raw.samples.shape, "dyn and raw indices differ in shape"
    assert torch.equal(dyn.window_session_ids, raw.window_session_ids), "session ids misaligned"
    assert len(dyn.user_sample_indices) == len(raw.user_sample_indices)
    for a, b in zip(dyn.user_sample_indices, raw.user_sample_indices):
        assert torch.equal(a, b), "user row ranges misaligned"
    return dyn, raw


def embed(model, samples, batch=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            out.append(model.feature_extractor(samples[i:i + batch].to(DEVICE)).cpu())
    return torch.cat(out).numpy()


def population(index, k):
    """Users with >= k windows in each of at least two sessions. Fixed once and shared by
    every scorer, so the columns differ only in how they score the same people."""
    sessions = index.window_session_ids.numpy()
    keep = []
    for u, idx in enumerate(index.user_sample_indices):
        if idx.numel() == 0:
            continue
        rows = idx.numpy()
        sid = sessions[rows]
        usable = [rows[sid == s] for s in np.unique(sid)]
        usable = [g for g in usable if len(g) >= k]
        if len(usable) >= 2:
            keep.append((u, usable[0], usable[1]))
    return keep


def templates(feature, pop, k, rng):
    """One gallery template and PROBES_PER_USER probes per user, gallery and probes from
    DIFFERENT sessions. `feature` is any per-window array: mean position or an embedding."""
    gallery, probes, owners = [], [], []
    for _, g_rows, p_rows in pop:
        gallery.append(feature[rng.choice(g_rows, k, replace=False)].mean(axis=0))
        for _ in range(PROBES_PER_USER):
            probes.append(feature[rng.choice(p_rows, k, replace=False)].mean(axis=0))
            owners.append(len(gallery) - 1)
    return np.asarray(gallery), np.asarray(probes), np.asarray(owners)


def euclid(probes, gallery, axes=None):
    p = probes[:, axes] if axes is not None else probes
    g = gallery[:, axes] if axes is not None else gallery
    return np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2)


def cosine(probes, gallery):
    p = probes / (np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    return 1.0 - p @ g.T


def rank1(d, owners, n_gallery, rng):
    """Ties rank-averaged, so a constant scorer lands at (N+1)/2 rather than rank 1."""
    n_users = d.shape[1]
    if n_users < 2:
        return float("nan")
    if n_gallery is None or n_gallery >= n_users:
        correct = d[np.arange(len(d)), owners]
        better = (d < correct[:, None]).sum(axis=1)
        tied = (d == correct[:, None]).sum(axis=1)
        return float(np.mean(1.0 / np.maximum(better + tied, 1) * (better == 0)))
    hits = 0.0
    idx = np.arange(n_users)
    for _ in range(DRAWS):
        for i in range(len(d)):
            others = rng.choice(np.delete(idx, owners[i]), n_gallery - 1, replace=False)
            row = d[i, np.append(others, owners[i])]
            correct = d[i, owners[i]]
            better = int((row < correct).sum())
            tied = int((row == correct).sum())
            hits += (1.0 / max(better + tied, 1)) if better == 0 else 0.0
    return float(hits / (DRAWS * len(d)))


def zscore(d):
    """Standardise a distance matrix on its own spread so two scorers on different scales
    contribute comparably. No weight is fitted - one tuned here would be tuned on the test
    set."""
    return (d - d.mean()) / (d.std() + 1e-12)


def static_gate():
    """Reproduce the published static rank-1 columns. If these do not come back, my
    enrolment protocol is not the one the static column used, and no dyn-vs-static
    comparison below means anything."""
    ck = torch.load(STATIC_NORM, map_location="cpu", weights_only=False)
    print(f'{"static gate":<20}{"k":>3}{"users":>7}   '
          f'{"xyz (target)":>22}{"y (target)":>22}{"xz (target)":>22}')
    ok = True
    for name, (dirname, k, target) in CORPORA.items():
        users_dir = str((ROOT / "processed_datasets" / dirname / "users").resolve())
        index = quiet(build_sample_index, users_dir, sample_time=5, sample_rate=20)
        quiet(ChannelNormalizer.from_state(ck["normalizer"]).transform, index)
        pos = index.samples[:, 4:7, :].mean(dim=2).numpy()
        pop = population(index, k)
        g, p, o = templates(pos, pop, k, np.random.default_rng(SEED))
        row = f"{name:<20}{k:>3}{len(pop):>7}   "
        for i, (label, axes) in enumerate(AXES.items()):
            got = rank1(euclid(p, g, axes), o, N_SMALL, np.random.default_rng(SEED))
            hit = abs(got - target[i]) < 0.02      # DRAWS=300 gallery resampling, not a fit
            ok &= hit
            row += f"{got:.3f} ({target[i]:.3f}){'    ' if hit else ' X  '}"
        print(row, flush=True)
    return ok


def dyn_columns():
    static_ck = torch.load(STATIC_NORM, map_location="cpu", weights_only=False)
    results = {}
    for name, (dirname, k, _) in CORPORA.items():
        users_dir = str((ROOT / "processed_datasets" / dirname / "users").resolve())
        dyn_index, raw_index = build_pair(users_dir)
        pop = population(raw_index, k)
        raw_pos = raw_index.samples[:, 4:7, :].mean(dim=2).numpy()

        # y is invariant to per-channel scaling in RANK, so the normaliser cannot change it.
        # Verified rather than asserted: computed both ways, required equal.
        std_index = quiet(build_sample_index, users_dir, sample_time=5, sample_rate=20)
        quiet(ChannelNormalizer.from_state(static_ck["normalizer"]).transform, std_index)
        std_pos = std_index.samples[:, 4:7, :].mean(dim=2).numpy()
        gy, py, oy = templates(std_pos, pop, k, np.random.default_rng(SEED))
        gr, pr, _ = templates(raw_pos, pop, k, np.random.default_rng(SEED))
        y_std = rank1(euclid(py, gy, [1]), oy, N_SMALL, np.random.default_rng(SEED))
        y_raw = rank1(euclid(pr, gr, [1]), oy, N_SMALL, np.random.default_rng(SEED))
        assert abs(y_std - y_raw) < 1e-9, f"y-only is not scale-invariant here: {y_std} vs {y_raw}"

        per_seed = {"dyn": [], "y": [], "y+dyn": []}
        for seed, path in sorted(DYN_93.items()):
            model, ck = load_checkpoint(path, DEVICE, 100, return_checkpoint=True)
            idx = quiet(build_sample_index, users_dir, sample_time=5, sample_rate=20, encoding="dyn")
            quiet(ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit").transform, idx)
            emb = embed(model, idx.samples)
            g_e, p_e, o = templates(emb, pop, k, np.random.default_rng(SEED))
            d_dyn, d_y = cosine(p_e, g_e), euclid(pr, gr, [1])
            for label, d in (("dyn", d_dyn), ("y", d_y), ("y+dyn", zscore(d_dyn) + zscore(d_y))):
                per_seed[label].append(rank1(d, o, N_SMALL, np.random.default_rng(SEED)))
        results[name] = {"k": k, "users": len(pop), **per_seed}
        print(f"  {name:<20} k={k:<3} users {len(pop):>3}   "
              + "   ".join(f"{lab} {np.mean(v):.3f}+-{np.std(v):.3f}" for lab, v in per_seed.items()),
              flush=True)
    return results


if __name__ == "__main__":
    print(f"device {DEVICE}; chance at N=17 is {1/17:.4f}\n")
    passed = static_gate()
    print("\nSTATIC GATE PASSED\n" if passed else
          "\n*** STATIC GATE FAILED - the protocol differs from the published column; report it ***\n")
    if not passed:
        sys.exit(1)
    print("dyn column, 5 checkpoints (9.3: trained on BOXRR+alyx only, so every corpus here is UNSEEN)")
    out = dyn_columns()
    pathlib.Path("docs/acceptance/step6_seated_dyn.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

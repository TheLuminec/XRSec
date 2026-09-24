"""
Questset cross-application scoring harness — the plumbing for arms A1/A2/A3.

    python docs/acceptance/questset_harness.py --gate
    python docs/acceptance/questset_harness.py --checkpoint <path.pth> [--encoding dyn]

Gallery = per-user template over game A; probe = individual game B windows;
rank-1 over a random gallery of N users, averaged over draws and over BOTH
ordered directions; reported per group and at N=17 (matched to Across-XR) and
N=30 (the reason this corpus was acquired).

NO GPU IS NEEDED. Every arm scores checkpoints that already exist, so this runs
on CPU. It is written to be usable the moment the weights are on this machine.

---------------------------------------------------------------------------
WHAT THE GATE COVERS, AND WHAT IT DOES NOT — read before trusting a number
---------------------------------------------------------------------------

This file **imports** `window_means`, `standardise` and `rank1` from
`questset_static_lookup.py` rather than reimplementing them. That is deliberate:
this project's rule is that a gate must be ONE implementation, because two
independently written comparators can disagree for reasons that have nothing to
do with the thing being compared. The cost is that the gate cannot re-validate
the shared primitives — it would be checking them against themselves.

So state the coverage honestly:

  COVERED by `--gate` (it reproduces `questset_static_lookup.json` exactly):
    - the population: which users are in which group, and that the identity key
      is (group, order, user) rather than the source's colliding `user` column
    - the pairing: which application is gallery and which is probe, and that
      both ordered directions are averaged
    - the gallery draw: N, the number of draws, and the rng seed
    - the assembly of results into per-group / per-axis / per-N cells

  NOT COVERED — validated elsewhere, named here so nobody assumes otherwise:
    - `rank1`, `standardise` and `window_means` themselves. Their own fixture
      gate in `questset_static_lookup.py` asserts the statistic at BOTH extremes
      (1.000 and 0.000) plus a known-answer fixture, which is what a shared
      primitive needs.
    - the model path. A checkpoint must first reproduce its OWN recorded figure
      to <1e-4 (`docs/acceptance/schach_ours_*_gate_cpu.json`); this harness
      REFUSES to emit a model number without that certificate — see below.
    - the resampling from the corpus's ~60 Hz to the pipeline's rate, which the
      model path needs and the static path does not. Flagged in the output.

---------------------------------------------------------------------------
TWO CORPUS FACTS THIS HARNESS ENCODES SO NOBODY REDISCOVERS THEM
---------------------------------------------------------------------------

1. **The identity key is (group, order, user).** The source's own `User` column
   runs 0..14 inside EVERY (group, order) block, so keying on it alone merges
   four different people into one identity. The converted corpus already encodes
   the triple as `g<G>o<O>u<UU>`; this harness never parses anything else.
2. **Five sessions run at ~112–116 Hz against a corpus otherwise at
   59.93 ± 0.28 Hz, and all five are in group 1** (`g1o1u07` both sessions,
   `g1o2u14` both, `g1o2u07` cooking only). `g1o2u07` therefore holds one 60 Hz
   and one 114 Hz session, so for that single identity a cross-application arm
   draws gallery and probe at different native rates — the only place resampling
   acts WITHIN a person. Both facts are reported in the output rather than left
   to be met in the residuals, because group 1 is one whole side of arm A2.

Per the standing rule adopted 2026-09-17: **this harness writes its result
artefact before anything reasons about it.** The JSON is written as soon as the
numbers exist, not at the end of an analysis.
"""

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

# ONE implementation of the shared primitives -- see the coverage note above.
from questset_static_lookup import (  # noqa: E402
    CORPUS, GROUPS, SEED, WINDOW_S, RATE_HZ, window_means, standardise, rank1,
)

GALLERY_SIZES = (17, 30)
DRAWS = 20
# The five sessions that are not at ~60 Hz, from docs/acceptance (manifest.json).
ODD_RATE_SESSIONS = {
    ("g1o1u07", "beat_saber"), ("g1o1u07", "cooking_simulator"),
    ("g1o2u07", "cooking_simulator"),
    ("g1o2u14", "beat_saber"), ("g1o2u14", "cooking_simulator"),
}
AXES = {"xyz": (0, 1, 2), "y only": (1,), "xz only": (0, 2)}


def load_group(gid, game_a, game_b, feature_fn):
    """Return {user: [feature, ...]} per game for one group, using feature_fn."""
    per_game = {game_a: {}, game_b: {}}
    for user_dir in sorted((CORPUS / "users").iterdir()):
        if not user_dir.name.startswith(f"g{gid}"):
            continue
        for game in (game_a, game_b):
            path = user_dir / f"{game}.csv"
            if path.exists():
                per_game[game][user_dir.name] = feature_fn(path)
    users = sorted(set(per_game[game_a]) & set(per_game[game_b]))
    return per_game, users


def score_group(per_game, users, game_a, game_b, axes, rng, standardise_features):
    """rank-1 averaged over both ordered directions, per gallery size."""
    flat = {u: per_game[game_a][u] + per_game[game_b][u] for u in users}
    n_a = {u: len(per_game[game_a][u]) for u in users}
    if standardise_features:
        flat, _, _ = standardise(flat)
    src = {u: flat[u][:n_a[u]] for u in users}
    dst = {u: flat[u][n_a[u]:] for u in users}

    out = {}
    for n_gal in GALLERY_SIZES:
        if n_gal > len(users):
            continue
        vals = []
        for gal_side, probe_side in ((src, dst), (dst, src)):
            gallery = {u: tuple(st.mean(w[i] for w in gal_side[u])
                                for i in range(len(gal_side[u][0]))) for u in users}
            vals.append(rank1(gallery, probe_side, axes, users, n_gal, DRAWS, rng))
        out[n_gal] = {"mean": sum(vals) / 2, "a_to_b": vals[0], "b_to_a": vals[1],
                      "chance": 1.0 / n_gal, "n_users": len(users)}
    return out


def run_static(rng):
    """The training-free mean-position cue, through this harness's own wiring."""
    results = {}
    for gid, (game_a, game_b) in GROUPS.items():
        per_game, users = load_group(gid, game_a, game_b, window_means)
        results[gid] = {"games": [game_a, game_b], "n_users": len(users), "axes": {}}
        for name, axes in AXES.items():
            results[gid]["axes"][name] = score_group(
                per_game, users, game_a, game_b, axes, rng, standardise_features=True)
    return results


def gate():
    """Reproduce questset_static_lookup.json through this harness's wiring."""
    ref_path = HERE / "questset_static_lookup.json"
    if not ref_path.exists():
        print(f"FAIL: no reference at {ref_path}; run questset_static_lookup.py first")
        return 1
    ref = json.load(ref_path.open())["groups"]

    got = run_static(random.Random(SEED))
    (HERE / "questset_harness_gate.json").write_text(json.dumps(got, indent=2))  # artefact FIRST

    worst, rows, failed = 0.0, [], 0
    for gid in sorted(ref):
        for name in AXES:
            for n in GALLERY_SIZES:
                key = str(n)
                if key not in ref[gid]["axes"][name]:
                    continue
                a = ref[gid]["axes"][name][key]["mean"]
                b = got[gid]["axes"][name][n]["mean"]
                d = abs(a - b)
                worst = max(worst, d)
                ok = d < 1e-12
                failed += (not ok)
                rows.append((gid, name, n, a, b, d, ok))

    print("GATE: harness static path against questset_static_lookup.json\n")
    print(f"  {'grp':>3} {'axis':9} {'N':>3} {'reference':>10} {'harness':>10} {'diff':>9}")
    for gid, name, n, a, b, d, ok in rows:
        print(f"  {gid:>3} {name:9} {n:>3} {a:10.4f} {b:10.4f} {d:9.1e} {'ok' if ok else 'FAIL'}")
    print(f"\n  cells {len(rows)}   failed {failed}   worst |diff| {worst:.1e}")

    if failed:
        print("\n  GATE FAILED. The harness builds a different population, pairing or draw")
        print("  than the committed lookup. No model number from it means anything.")
        return 1
    print("\n  GATE PASSED - population, identity key, pairing, both directions,")
    print("  gallery draw and assembly all match. See the coverage note in this")
    print("  file's docstring for what this does NOT establish.")
    return 0


# The Questset registration's checkpoint gate: the checkpoint reproduces its OWN recorded figure
# on its OWN evaluation users to < 1e-4 before any number is quoted, and the gap is written down.
QUESTSET_GATE_TOLERANCE = 1e-4


def embed_corpus(ck: dict, model, device):
    """Every Questset window through the PIPELINE's own sampler, encoder and normaliser at the
    checkpoint's settings. Questset is in no checkpoint's normaliser, so the pipeline's unseen
    policy applies (target_fit: statistics fitted on Questset itself, unsupervised) - the same
    policy for every arm, recorded in the output."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "model"))
    from dataset import SampleDataset, SampleIndex
    from normalization import ChannelNormalizer
    from across_xr_alignment import quiet, embed

    es = ck["eval_split"]
    ds = quiet(SampleDataset, str(CORPUS / "users"), sample_time=int(es["sample_time"]),
               sample_rate=int(es["sample_rate"]), channels=ck.get("channels", "full"),
               resample=es.get("resample", "nearest"), window_stride=es.get("window_stride"))
    index = SampleIndex(ds, encoding=es.get("encoding", "raw"))
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if normalizer.enabled:
        quiet(normalizer.transform, index)
    sessions = index.window_session_ids.numpy()
    window_user = np.empty(index.sample_count, dtype=object)
    window_app = np.empty(index.sample_count, dtype=object)
    for user_dir, rows in zip(ds.user_dirs, index.user_sample_indices):
        name = Path(user_dir).name
        csvs = sorted(f for f in (CORPUS / "users" / name).iterdir() if f.suffix == ".csv")
        apps = [f.stem for f in csvs]
        rows = rows.numpy()
        assert set(sessions[rows].tolist()) <= set(range(len(apps))), (name, apps)
        window_user[rows] = name
        window_app[rows] = np.array(apps, dtype=object)[sessions[rows]]
    assert len(ds.user_dirs) == 60, f"expected 60 Questset users, loaded {len(ds.user_dirs)}"
    return embed(model, index.samples, device), window_user, window_app, dict(normalizer.unseen_datasets)


def score_model_group(emb, window_user, window_app, users, game_a, game_b, rng):
    """Cross-application rank-1, the Across-XR A1 rule (template = renormalised mean of L2-normalised
    window embeddings over game A; each game-B window a probe; cosine; ties rank-averaged), over a
    random gallery of N users drawn by THIS harness's gated draw, averaged over draws and over both
    ordered directions.

    PRIMARY, fixed 2026-09-24 before any model number existed: the mean over users of per-user
    rank-1 (the Across-XR A1 definition). `window_pooled` (every probe window weighted equally, the
    static lookup's weighting) is reported beside it and is not the registered figure."""
    import numpy as np
    from across_xr_alignment import centroids, rank1_per_user

    def rows(u, g):
        return np.flatnonzero((window_user == u) & (window_app == g))

    out = {}
    for n_gal in GALLERY_SIZES:
        if n_gal > len(users):
            continue
        per_dir = []
        for gal_game, probe_game in ((game_a, game_b), (game_b, game_a)):
            vals, pooled = [], []
            for _ in range(DRAWS):
                pool = rng.sample(users, n_gal)
                gallery = centroids(emb, [rows(u, gal_game) for u in pool])
                p_rows = [rows(u, probe_game) for u in pool]
                probe_user = np.concatenate([np.full(len(r), i) for i, r in enumerate(p_rows)])
                per_user = rank1_per_user(gallery, emb[np.concatenate(p_rows)], probe_user, n_gal)
                vals.append(float(per_user.mean()))
                pooled.append(float(np.average(per_user, weights=[len(r) for r in p_rows])))
            per_dir.append((float(np.mean(vals)), float(np.mean(pooled))))
        out[n_gal] = {"mean": (per_dir[0][0] + per_dir[1][0]) / 2,
                      "a_to_b": per_dir[0][0], "b_to_a": per_dir[1][0],
                      "window_pooled": (per_dir[0][1] + per_dir[1][1]) / 2,
                      "chance": 1.0 / n_gal, "n_users": len(users)}
    return out


def run_model(checkpoint: Path, encoding: str, device_name: str = "cpu", out: Path | None = None):
    """Score a checkpoint. Refuses unless it first reproduces its own recorded figure to < 1e-4."""
    import torch
    from across_xr_alignment import gate as checkpoint_gate

    if not checkpoint.exists():
        print(f"CHECKPOINT NOT ON THIS MACHINE: {checkpoint}")
        return 3
    device = torch.device(device_name)
    g, model, ck = checkpoint_gate(str(checkpoint), device)
    if not g.get("passed") or g.get("gap") is None or g["gap"] >= QUESTSET_GATE_TOLERANCE:
        print(f"REFUSING: checkpoint gate gap {g.get('gap')} on {device} is not < {QUESTSET_GATE_TOLERANCE} "
              f"({g.get('reason', '')}). Score on the device that trained it.")
        return 2
    rec_enc = ck["eval_split"].get("encoding", "raw")
    if rec_enc != encoding:
        print(f"REFUSING: checkpoint encoding is {rec_enc}, --encoding says {encoding}")
        return 2
    emb, window_user, window_app, unseen = embed_corpus(ck, model, device)
    rng = random.Random(SEED)
    groups = {}
    for gid, (game_a, game_b) in GROUPS.items():
        users = sorted({u for u in set(window_user.tolist()) if u.startswith(f"g{gid}")
                        and ((window_user == u) & (window_app == game_a)).any()
                        and ((window_user == u) & (window_app == game_b)).any()})
        groups[gid] = {"games": [game_a, game_b], "n_users": len(users),
                       "cells": score_model_group(emb, window_user, window_app, users, game_a, game_b, rng)}
    result = {"checkpoint": str(checkpoint), "device": str(device), "encoding": encoding,
              "gate": {k: v for k, v in g.items() if k != "exclude_users_seen"},
              "unseen_policy": unseen, "windows": int(len(emb)), "draws": DRAWS, "seed": SEED,
              "primary": "per-user mean rank-1 (Across-XR A1 rule); window_pooled reported beside it",
              "groups": groups}
    out = out or HERE / f"questset_model_{checkpoint.parent.parent.name}.json"
    Path(out).write_text(json.dumps(result, indent=1))          # artefact FIRST
    print(f"wrote {out}")
    for gid, r in groups.items():
        for n, c in r["cells"].items():
            print(f"  group {gid} {r['games'][0]}<->{r['games'][1]} N={n}: rank-1 {c['mean']:.4f} "
                  f"(pooled {c['window_pooled']:.4f}, chance {c['chance']:.3f}, users {c['n_users']})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true",
                    help="reproduce the committed static lookup through this harness")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--encoding", default="dyn", choices=("dyn", "raw"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"corpus {CORPUS}   windows {WINDOW_S}s at {RATE_HZ}Hz   seed {SEED}   draws {DRAWS}")
    print(f"NOTE: {len(ODD_RATE_SESSIONS)} sessions run at ~112-116Hz, ALL in group 1;")
    print("      g1o2u07 holds one 60Hz and one 114Hz session, so its own cross-application")
    print("      pair is not internally rate-comparable. Carry this into arm A2.\n")

    if args.gate:
        return gate()
    if args.checkpoint:
        return run_model(args.checkpoint, args.encoding, args.device, args.out)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

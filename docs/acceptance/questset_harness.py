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


def model_gate_certificates():
    return sorted(HERE.glob("schach_ours_*_gate_cpu.json"))


def run_model(checkpoint: Path, encoding: str):
    """Score a checkpoint. Refuses without a checkpoint gate certificate."""
    certs = model_gate_certificates()
    if not certs:
        print("REFUSING: no checkpoint gate certificate found in docs/acceptance/.")
        print("A checkpoint must reproduce its OWN recorded figure to <1e-4 before any")
        print("Questset number is quoted from it. That is the rule that voided the step-6")
        print("columns twice.")
        return 2
    if not checkpoint.exists():
        print(f"CHECKPOINT NOT ON THIS MACHINE: {checkpoint}")
        print()
        print("The programme checkpoints live on the node that trained them and have not")
        print("been replicated here (verified 2026-09-17). Until they are, no Questset arm")
        print("can run anywhere but that node - which is the exposure that cost 36 hours")
        print("when the Miami node was lost. Replicate them, then re-run this command.")
        print()
        print(f"Gate certificates present for: {[c.name for c in certs]}")
        return 3
    print("Checkpoint present; model scoring path is not yet implemented.")
    print("Implement embed_windows() against model/load_checkpoint + the pipeline's")
    print("Sampler at the checkpoint's own sample_time/sample_rate/encoding, then")
    print("reuse score_group() unchanged - the wiring around it is already gated.")
    return 4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true",
                    help="reproduce the committed static lookup through this harness")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--encoding", default="dyn", choices=("dyn", "raw"))
    args = ap.parse_args()

    print(f"corpus {CORPUS}   windows {WINDOW_S}s at {RATE_HZ}Hz   seed {SEED}   draws {DRAWS}")
    print(f"NOTE: {len(ODD_RATE_SESSIONS)} sessions run at ~112-116Hz, ALL in group 1;")
    print("      g1o2u07 holds one 60Hz and one 114Hz session, so its own cross-application")
    print("      pair is not internally rate-comparable. Carry this into arm A2.\n")

    if args.gate:
        return gate()
    if args.checkpoint:
        return run_model(args.checkpoint, args.encoding)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

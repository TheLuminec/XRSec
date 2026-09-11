"""
Aggregate the per-seed alignment certificates and score the registered verdicts.

Two kinds of comparison, both over the same 17 test users:

  within a checkpoint   A2 - A1, A2' - A1, ... : per-user paired differences, pooled over
                        seeds, cluster-bootstrapped over users; plus the per-seed means so the
                        seed-spread rule (A2 - A1 means within 0.05 of each other) is checked
  across checkpoints    C2-hi - Z-676, C2-lo - A1, Z-676 - A1, C2-hi - C2-lo: the SAME users'
                        per-user A1 accuracy under two different models, paired by user and
                        seed, bootstrapped over users

    python docs/acceptance/across_xr_alignment_aggregate.py \
        --zero-shot across_xr_alignment_seed1.json across_xr_alignment_seed2.json ... \
        [--c1 ...json] [--z676 ...json] [--c2hi ...json] [--c2lo ...json] [--out ...json]

Every figure is rank-1 at N=17, single 10 s window; nothing here re-embeds anything.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
N_BOOT = 10000
ARMS = ("A0", "A1", "A2", "A2prime", "A2null", "A2full")


def load(paths):
    out = []
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        for s in d["seeds"]:
            if s.get("gate", {}).get("passed") is not True:
                raise SystemExit(f"{p}: gate not passed - nothing from it is quotable")
            out.append(s)
    return out


def per_user(seed, arm):
    return np.asarray(seed["arms"][arm]["per_user"], dtype=float)


def boot(values, rng):
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled_user_vector(seeds, arm):
    """Per-user accuracy averaged over seeds: the unit the bootstrap resamples is still the
    user, so seeds are averaged inside each user rather than treated as more users."""
    return np.mean([per_user(s, arm) for s in seeds], axis=0)


def summarise(name, seeds, rng):
    rows = {}
    for arm in ARMS:
        means = [s["arms"][arm]["mean"] for s in seeds]
        mean, lo, hi = boot(pooled_user_vector(seeds, arm), rng)
        rows[arm] = {"mean": mean, "ci95": [lo, hi], "per_seed": means,
                     "seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                     "sequence_10min": float(np.mean([s["arms"][arm]["sequence_10min"] for s in seeds]))}
        print(f"  {name} {arm:8s} {mean:.3f} [{lo:.3f}, {hi:.3f}]  seeds {' '.join(f'{m:.3f}' for m in means)}"
              f"  10-min {rows[arm]['sequence_10min']:.3f}")
    contrasts = {}
    for a, b in (("A2prime", "A1"), ("A2", "A1"), ("A2", "A2prime"), ("A2null", "A1"), ("A2full", "A2")):
        diff = pooled_user_vector(seeds, a) - pooled_user_vector(seeds, b)
        mean, lo, hi = boot(diff, rng)
        per_seed = [s["arms"][a]["mean"] - s["arms"][b]["mean"] for s in seeds]
        contrasts[f"{a}-{b}"] = {"mean": mean, "ci95": [lo, hi], "per_seed": per_seed,
                                 "seed_range": float(max(per_seed) - min(per_seed))}
        print(f"  {name} {a}-{b}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  per seed {' '.join(f'{v:+.3f}' for v in per_seed)}"
              f"  range {contrasts[f'{a}-{b}']['seed_range']:.3f}")
    m = [s["m_curve_valid_n9"] for s in seeds]
    print(f"  {name} m-curve flat in {sum(bool(x['flat_below_0.05']) for x in m)}/{len(m)} seeds; m* {[x['m_star'] for x in m]}")
    return {"arms": rows, "contrasts": contrasts, "m_curves": m, "n_seeds": len(seeds)}


def cross(name, left, right, rng, arm="A1"):
    """left - right on the same users, paired by user and by seed order."""
    k = min(len(left), len(right))
    diff = np.mean([per_user(left[i], arm) - per_user(right[i], arm) for i in range(k)], axis=0)
    mean, lo, hi = boot(diff, rng)
    print(f"  {name}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] over {k} paired seed(s)")
    return {"mean": mean, "ci95": [lo, hi], "seeds": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-shot", nargs="+", required=True)
    for arm in ("c1", "z676", "c2hi", "c2lo"):
        ap.add_argument(f"--{arm}", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "across_xr_alignment_aggregate.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(67)
    out = {}
    zs = load(args.zero_shot)
    print(f"zero-shot: {len(zs)} seeds")
    out["zero_shot"] = summarise("zero-shot", zs, rng)
    groups = {"C1": load(args.c1), "Z676": load(args.z676), "C2hi": load(args.c2hi), "C2lo": load(args.c2lo)}
    for name, seeds in groups.items():
        if seeds:
            print(f"{name}: {len(seeds)} seed(s)")
            out[name] = summarise(name, seeds, rng)
    out["cross"] = {}
    if groups["C2hi"] and groups["Z676"]:
        out["cross"]["C2hi-Z676"] = cross("C2-hi - Z-676 (A1, same users)", groups["C2hi"], groups["Z676"], rng)
    if groups["Z676"]:
        out["cross"]["Z676-A1"] = cross("Z-676 - zero-shot 4096 (A1)", groups["Z676"], zs, rng)
    if groups["C2lo"]:
        out["cross"]["C2lo-A1"] = cross("C2-lo - zero-shot 4096 (A1)", groups["C2lo"], zs, rng)
    if groups["C2hi"] and groups["C2lo"]:
        out["cross"]["C2hi-C2lo"] = cross("C2-hi - C2-lo (A1)", groups["C2hi"], groups["C2lo"], rng)
    if groups["C1"]:
        out["cross"]["C1-A1"] = cross("C1 - zero-shot 4096 (A1)", groups["C1"], zs, rng)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

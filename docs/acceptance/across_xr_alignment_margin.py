"""
Amendment 7: identity_margin 0.1 / identity_scale 15 against the 0.35 / 30 default, paired on
the 17 users - M-zero against the zero-shot arm, M-C2-lo against C2-lo - A1 (cross-application)
as the registered contrast, A0 and the 10-minute sequence reported beside it. Seeds are
averaged inside each user before the bootstrap. Bands -0.02..+0.04; above +0.04 the lever
transfers; below -0.02 the default stands at this identity count.

    python docs/acceptance/across_xr_alignment_margin.py \
        --base-zero seed1.json seed2.json seed3.json --m-zero m1.json [...] \
        [--base-c2lo ... --m-c2lo ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from across_xr_alignment_aggregate import boot, load, per_user  # noqa: E402


def contrast(m, base, arm, rng):
    a = np.mean([per_user(s, arm) for s in m], axis=0)
    b = np.mean([per_user(s, arm) for s in base], axis=0)
    mean, lo, hi = boot(a - b, rng)
    return {"m_mean": float(a.mean()), "base_mean": float(b.mean()), "diff": mean, "lo": lo, "hi": hi,
            "m_per_seed": [s["arms"][arm]["mean"] for s in m], "n_m": len(m), "n_base": len(base)}


def verdict(c):
    if c["lo"] > 0.04:
        return "above the band: the lever transfers to cross-application rank-1 by more than its in-domain measurement"
    if c["hi"] < -0.02:
        return "below the band: a margin tuned at 419 identities is wrong at this count; the default stands"
    if c["lo"] >= -0.02 and c["hi"] <= 0.04:
        return "inside the band -0.02..+0.04"
    return "interval spans a registered edge: unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-zero", nargs="+", required=True)
    ap.add_argument("--m-zero", nargs="+", required=True)
    ap.add_argument("--base-c2lo", nargs="*", default=[])
    ap.add_argument("--m-c2lo", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "across_xr_alignment_margin.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(67)
    out = {}
    for fam, base_paths, m_paths in (("zero-shot", args.base_zero, args.m_zero), ("C2-lo", args.base_c2lo, args.m_c2lo)):
        if not m_paths:
            continue
        base, m = load(base_paths), load(m_paths)
        for s in m:
            g = s["gate"]
            print(f"  {fam} 0.1/15 gate: recorded {g['recorded']:.6f} rescored {g['rescored']:.6f} gap {g['gap']:.1e}")
        rec = {arm: contrast(m, base, arm, rng) for arm in ("A1", "A0")}
        rec["sequence_10min_diff"] = float(np.mean([s["arms"]["A1"]["sequence_10min"] for s in m]) - np.mean([s["arms"]["A1"]["sequence_10min"] for s in base]))
        rec["verdict_A1"] = verdict(rec["A1"])
        c = rec["A1"]
        print(f"  {fam} A1: 0.1/15 {c['m_mean']:.3f} (seeds {' '.join(f'{v:.3f}' for v in c['m_per_seed'])}) - 0.35/30 {c['base_mean']:.3f} "
              f"= {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] -> {rec['verdict_A1']}; A0 diff {rec['A0']['diff']:+.3f}; 10-min diff {rec['sequence_10min_diff']:+.3f}")
        out[fam] = rec
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

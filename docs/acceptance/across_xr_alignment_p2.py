"""
P2 (Amendment 6): raw minus dyn on the same cells, paired on the 17 users.

Two contrasts per arm family, both against the registered bands:
  A1 (cross-application) raw - dyn : the anthropometric contribution across applications;
                                     band +0.00..+0.06, falsifier below -0.03
  A0 (within-application) raw - dyn: carries the within-application placement cue
                                     (P=0.7525) and is reported, never quoted as biometric;
                                     registered to exceed the A1 gain
Seeds are averaged inside each user before the bootstrap. Every raw row's
position_lookup_auc is printed beside it from the gate record.

    python docs/acceptance/across_xr_alignment_p2.py \
        --dyn-zero seed1.json seed2.json seed3.json --raw-zero r1.json [r2.json r3.json] \
        [--dyn-c2lo ... --raw-c2lo ...] [--out ...]
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


def contrast(name, raw, dyn, arm, rng):
    r = np.mean([per_user(s, arm) for s in raw], axis=0)
    d = np.mean([per_user(s, arm) for s in dyn], axis=0)
    m, lo, hi = boot(r - d, rng)
    per_seed = [s["arms"][arm]["mean"] for s in raw]
    print(f"  {name} {arm}: raw {r.mean():.3f} (seeds {' '.join(f'{v:.3f}' for v in per_seed)}) - dyn {d.mean():.3f} "
          f"= {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (raw {len(raw)} seed(s), dyn {len(dyn)})")
    return {"raw_mean": float(r.mean()), "dyn_mean": float(d.mean()), "diff": m, "lo": lo, "hi": hi,
            "raw_per_seed": per_seed, "n_raw": len(raw), "n_dyn": len(dyn)}


def verdict_a1(c):
    if c["lo"] > 0.06:
        return "above the band: height is worth more than registered"
    if c["hi"] < -0.03:
        return "FALSIFIER: the frame problem costs more than height returns; dyn stays the encoding"
    if c["lo"] >= -0.03 and c["hi"] <= 0.06:
        return "inside the band +0.00..+0.06 (interval within the named region)"
    return "interval spans a registered edge: unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn-zero", nargs="+", required=True)
    ap.add_argument("--raw-zero", nargs="+", required=True)
    ap.add_argument("--dyn-c2lo", nargs="*", default=[])
    ap.add_argument("--raw-c2lo", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "across_xr_alignment_p2.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(67)
    out = {}
    for fam, dyn_paths, raw_paths in (("zero-shot", args.dyn_zero, args.raw_zero), ("C2-lo", args.dyn_c2lo, args.raw_c2lo)):
        if not raw_paths:
            continue
        dyn, raw = load(dyn_paths), load(raw_paths)
        for s in raw:
            g = s["gate"]
            print(f"  {fam} raw gate: recorded {g['recorded']:.6f} rescored {g['rescored']:.6f} gap {g['gap']:.1e} "
                  f"position_lookup {g.get('position_lookup_recorded')}")
        a1 = contrast(fam, raw, dyn, "A1", rng)
        a0 = contrast(fam, raw, dyn, "A0", rng)
        seq = float(np.mean([s["arms"]["A1"]["sequence_10min"] for s in raw]) - np.mean([s["arms"]["A1"]["sequence_10min"] for s in dyn]))
        rec = {"A1": a1, "A0": a0, "sequence_10min_diff": seq, "verdict_A1": verdict_a1(a1),
               "A0_gain_exceeds_A1_gain": bool(a0["diff"] > a1["diff"])}
        print(f"  {fam}: verdict A1 -> {rec['verdict_A1']}; A0 gain {a0['diff']:+.3f} {'>' if rec['A0_gain_exceeds_A1_gain'] else '<='} A1 gain "
              f"{a1['diff']:+.3f} (placement reading {'holds' if rec['A0_gain_exceeds_A1_gain'] else 'does not hold'}); 10-min diff {seq:+.3f}")
        out[fam] = rec
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

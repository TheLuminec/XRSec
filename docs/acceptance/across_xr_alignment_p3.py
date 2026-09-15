"""
P3 (Amendment 4): leave-one-application-out, unseen users - the aggregation.

Each P3 run holds out one application X. Its unit is the eight ordered cells involving X,
paired on the same 17 users against the same cells of Z-676 (no exposure) and C2-hi (full
exposure), then pooled over X (per-user differences averaged over held-out applications,
then the user bootstrap). The twelve non-X cells against C2-hi's are the dose control (P3
has ~11.6% Across-XR windows against C2-hi's 14.1%), a free measurement of what 20% less
in-domain data is worth at fixed exposure. Nothing here re-embeds anything: it reads the
per-cell per-user arrays the harness records.

    python docs/acceptance/across_xr_alignment_p3.py --z676 z676.json --c2hi c2hi.json \
        --p3 superhot_vr=... half_life_alyx=... beat_saber=... synth_riders=... social_vr=... [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from across_xr_alignment_aggregate import boot, load  # noqa: E402

GAMES = ("superhot_vr", "half_life_alyx", "beat_saber", "synth_riders", "social_vr")


def cell_user(seed, arm, cell):
    return np.asarray(seed["arms"][arm]["per_cell_user"][cell], dtype=float)


def cells_involving(x):
    return [f"{a}->{b}" for a in GAMES for b in GAMES if a != b and (a == x or b == x)]


def cells_not_involving(x):
    return [f"{a}->{b}" for a in GAMES for b in GAMES if a != b and a != x and b != x]


def mean_over(seed, arm, cells):
    return np.mean([cell_user(seed, arm, c) for c in cells], axis=0)


def ci(values, rng):
    m, lo, hi = boot(values, rng)
    return {"mean": m, "lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z676", required=True)
    ap.add_argument("--c2hi", required=True)
    ap.add_argument("--p3", nargs="+", required=True, help="X=path pairs")
    ap.add_argument("--out", default=str(HERE / "across_xr_alignment_p3.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(67)
    z, c = load([args.z676])[0], load([args.c2hi])[0]
    runs = {}
    for item in args.p3:
        x, path = item.split("=", 1)
        assert x in GAMES, x
        runs[x] = load([path])[0]
    out = {"per_x": {}, "pooled": {}}
    vs_z, vs_c, ctrl, a1x = [], [], [], []
    for x, s in runs.items():
        xc, nx = cells_involving(x), cells_not_involving(x)
        assert len(xc) == 8 and len(nx) == 12
        px, zx, cx = mean_over(s, "A1", xc), mean_over(z, "A1", xc), mean_over(c, "A1", xc)
        pn, cn = mean_over(s, "A1", nx), mean_over(c, "A1", nx)
        rec = {"P3_x": ci(px, rng), "Z676_x": float(zx.mean()), "C2hi_x": float(cx.mean()),
               "P3-Z676_x": ci(px - zx, rng), "P3-C2hi_x": ci(px - cx, rng),
               "control_P3-C2hi_nonx": ci(pn - cn, rng),
               "A2prime-A1": s["paired"]["A2prime-A1"], "A2-A1": s["paired"]["A2-A1"],
               "gate": s["gate"]}
        out["per_x"][x] = rec
        vs_z.append(px - zx); vs_c.append(px - cx); ctrl.append(pn - cn); a1x.append(px)
        print(f"  held out {x:15s} X-cells: P3 {px.mean():.3f}  Z-676 {zx.mean():.3f}  C2-hi {cx.mean():.3f} | "
              f"P3-Z {rec['P3-Z676_x']['mean']:+.3f} [{rec['P3-Z676_x']['lo']:+.3f}, {rec['P3-Z676_x']['hi']:+.3f}]  "
              f"P3-C2hi {rec['P3-C2hi_x']['mean']:+.3f}  control(non-X) {rec['control_P3-C2hi_nonx']['mean']:+.3f} "
              f"[{rec['control_P3-C2hi_nonx']['lo']:+.3f}, {rec['control_P3-C2hi_nonx']['hi']:+.3f}]  "
              f"A2'-A1 {rec['A2prime-A1']['mean']:+.3f}")
    for name, arr in (("P3-Z676_x_pooled", vs_z), ("P3-C2hi_x_pooled", vs_c), ("control_nonx_pooled", ctrl),
                      ("P3_x_pooled", a1x)):
        out["pooled"][name] = {**ci(np.mean(arr, axis=0), rng), "n_x": len(arr)}
        r = out["pooled"][name]
        print(f"  {name}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] over {len(arr)} held-out applications")
    pz = out["pooled"]["P3-Z676_x_pooled"]
    if pz["lo"] > 0.03:
        verdict = "HEADLINE: exposure to four applications carries to an unseen fifth (interval lower bound above +0.03)"
    elif pz["hi"] <= 0.0:
        verdict = "FALSIFIER: exposure to other applications does not carry to a new one; the in-set gain is strictly in-set"
    elif pz["mean"] <= 0.02:
        verdict = "in the named (0, +0.02] region: unresolved"
    else:
        verdict = "inside +0.02..+0.07 at the mean but the interval's lower bound is not above +0.03: the headline is not made"
    out["verdict"] = verdict
    print("  verdict:", verdict)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

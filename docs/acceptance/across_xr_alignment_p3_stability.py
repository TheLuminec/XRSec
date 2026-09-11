"""P3 seed stability (Amendment 4, third addendum): the two re-seeded applications, each
seed's X-cell contrast against Z-676, the seed range, and the pooled five-application
interval with the re-seeded applications averaged over seeds INSIDE each user (a seed is
never an extra person). The +0.030 headline threshold is unchanged by rule."""
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from across_xr_alignment_aggregate import boot, load
from across_xr_alignment_p3 import GAMES, cells_involving, mean_over

rng = np.random.default_rng(67)
z = load([HERE / "across_xr_alignment_z676_seed1.json"])[0]
c = load([HERE / "across_xr_alignment_c2hi_seed1.json"])[0]
seeds = {x: [load([HERE / f"across_xr_alignment_p3_{x}.json"])[0]] for x in GAMES}
for x in ("synth_riders", "social_vr"):
    seeds[x].append(load([HERE / f"across_xr_alignment_p3_{x}_seed2.json"])[0])

out = {"per_x": {}, "pooled": {}}
diffs, uncovered = [], []
for x in GAMES:
    xc = cells_involving(x)
    per_seed = [float((mean_over(s, "A1", xc) - mean_over(z, "A1", xc)).mean()) for s in seeds[x]]
    avg = np.mean([mean_over(s, "A1", xc) for s in seeds[x]], axis=0) - mean_over(z, "A1", xc)
    m, lo, hi = boot(avg, rng)
    out["per_x"][x] = {"P3-Z676_per_seed": per_seed, "seed_range": (max(per_seed) - min(per_seed)) if len(per_seed) > 1 else None,
                       "seed_avg": {"mean": m, "lo": lo, "hi": hi}, "n_seeds": len(per_seed)}
    diffs.append(avg)
    if x in ("superhot_vr", "synth_riders", "social_vr"):
        uncovered.append(avg)
    print(f"  {x:15s} P3-Z676 per seed {' / '.join(f'{v:+.3f}' for v in per_seed)}"
          f"{'  range %.3f' % out['per_x'][x]['seed_range'] if len(per_seed) > 1 else ''}  seed-avg {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for name, arr in (("pooled_five", diffs), ("uncovered_triple", uncovered)):
    m, lo, hi = boot(np.mean(arr, axis=0), rng)
    out["pooled"][name] = {"mean": m, "lo": lo, "hi": hi}
    print(f"  {name}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (headline needs lower bound > +0.030: {'MADE' if lo > 0.03 else 'NOT MADE'})")
(HERE / "across_xr_alignment_p3_stability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

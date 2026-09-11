"""Amendment 5: C2-lo-half - C2-lo on A1, paired on the 17 users (the half arm's per-user
accuracy against the mean over C2-lo's three seeds per user), user bootstrap. Registered:
within +-0.03 -> dose is not binding in this range; below -0.03 -> dose binds."""
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from across_xr_alignment_aggregate import boot, load, per_user

rng = np.random.default_rng(67)
half = load([HERE / "across_xr_alignment_c2lohalf_seed1.json"])[0]
c2lo = load([HERE / f"across_xr_alignment_c2lo_seed{s}.json" for s in (1, 2, 3)])
h = per_user(half, "A1")
c = np.mean([per_user(s, "A1") for s in c2lo], axis=0)
m, lo, hi = boot(h - c, rng)
seq = half["arms"]["A1"]["sequence_10min"] - float(np.mean([s["arms"]["A1"]["sequence_10min"] for s in c2lo]))
verdict = ("within +-0.03: dose is not binding in this range" if lo >= -0.03 and hi <= 0.03
           else "interval extends below -0.03: dose may bind - read C2-hi/C2-lo as scale minus a dose cost" if lo < -0.03 and hi <= 0.03
           else "interval extends above +0.03: less in-domain data helps - flagged for a seed" if hi > 0.03 and lo >= -0.03
           else "interval spans both registered edges: unresolved")
print(f"C2-lo-half - C2-lo (A1, paired on 17 users, half vs mean of 3 seeds): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]; 10-min {seq:+.3f}")
print("verdict:", verdict)
(HERE / "across_xr_alignment_dose.json").write_text(json.dumps(
    {"half_minus_c2lo_A1": {"mean": m, "lo": lo, "hi": hi}, "half_A1": half["arms"]["A1"]["mean"],
     "c2lo_A1_3seeds": float(c.mean()), "sequence_10min_delta": seq, "verdict": verdict}, indent=1), encoding="utf-8")

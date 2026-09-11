"""P3's registered coverage split (Amendment 4, second addendum): P3 - Z-676 on the held-out
cells pooled over the pretraining-COVERED pair {Beat Saber, Alyx} and the UNCOVERED triple
{Superhot, Synth Riders, Social VR}. The boundary claim is decided on the triple."""
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from across_xr_alignment_aggregate import boot, load
from across_xr_alignment_p3 import cells_involving, mean_over

COVERED = ("beat_saber", "half_life_alyx")
UNCOVERED = ("superhot_vr", "synth_riders", "social_vr")
rng = np.random.default_rng(67)
z = load([HERE / "across_xr_alignment_z676_seed1.json"])[0]
c = load([HERE / "across_xr_alignment_c2hi_seed1.json"])[0]
out = {}
for name, group in (("covered", COVERED), ("uncovered", UNCOVERED)):
    dz, dc = [], []
    for x in group:
        s = load([HERE / f"across_xr_alignment_p3_{x}.json"])[0]
        xc = cells_involving(x)
        dz.append(mean_over(s, "A1", xc) - mean_over(z, "A1", xc))
        dc.append(mean_over(s, "A1", xc) - mean_over(c, "A1", xc))
    mz, loz, hiz = boot(np.mean(dz, axis=0), rng)
    mc, loc, hic = boot(np.mean(dc, axis=0), rng)
    out[name] = {"apps": group, "P3-Z676": [mz, loz, hiz], "P3-C2hi": [mc, loc, hic]}
    print(f"{name:9s} {group}: P3-Z676 {mz:+.3f} [{loz:+.3f}, {hiz:+.3f}]   P3-C2hi {mc:+.3f} [{loc:+.3f}, {hic:+.3f}]")
lo = out["uncovered"]["P3-Z676"][1]
print("boundary claim (uncovered triple, lower bound > +0.03):", "MADE" if lo > 0.03 else "NOT MADE")
(HERE / "across_xr_alignment_p3_split.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

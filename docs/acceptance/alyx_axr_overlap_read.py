"""
Read alyx_axr_overlap_REGISTERED.md. Committed before any no-alyx run exists.

Per arm (zero-shot, C2-lo): A1 without alyx minus A1 with alyx, per user (Across-XR users 32-48), each side
averaged over its seeds, user bootstrap (across_xr_alignment_aggregate.boot, one implementation). Both sides
must come from the same harness pass: the "with" side is the re-scored originals, not the committed files.
Descriptive split: cells involving half_life_alyx against cells that do not.

    python docs/acceptance/alyx_axr_overlap_read.py [--dir docs/acceptance] [--out ...]
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
from across_xr_alignment_p3 import GAMES, cells_involving, cells_not_involving, mean_over  # noqa: E402

REGIONS = [(-np.inf, -0.03, "alyx pretraining CONTRIBUTES: overlap not excluded, nor content (Half-Life); read the per-cell split"),
           (-0.03, 0.03, "BAND: the result does not depend on Who-is-Alyx; overlap cannot inflate it by more than 0.03"),
           (0.03, np.inf, "removing alyx HELPS: unexpected, report as such")]
assert REGIONS[0][0] == -np.inf and REGIONS[-1][1] == np.inf and all(a[1] == b[0] for a, b in zip(REGIONS, REGIONS[1:]))
ALL = [f"{a}->{b}" for a in GAMES for b in GAMES if a != b]


def verdict(r):
    hit = [m for a, b, m in REGIONS if r["lo"] < b and r["hi"] >= a]
    at = [m for a, b, m in REGIONS if a <= r["mean"] < b]
    return f"interval inside one region: {hit[0]}" if len(hit) == 1 else f"interval spans {len(hit)} regions; the mean sits in: {at[0]}"


def ci(v, rng):
    m, lo, hi = boot(v, rng)
    return {"mean": m, "lo": lo, "hi": hi}


def result_files(d, arm, kind):
    # gate certificates (*_gate.json) sit beside the results and match the same glob (Amendment 2)
    return sorted(f for f in d.glob(f"across_xr_alignment_{arm}_{kind}_seed*.json") if not f.name.endswith("_gate.json"))


def arm_vectors(seeds, cells):
    return np.mean([mean_over(s, "A1", cells) for s in seeds], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--out", default=str(HERE / "alyx_axr_overlap_read.json"))
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    rng = np.random.default_rng(67)
    out = {"rows": {}, "verdicts": {}, "descriptive": {}, "levels": {}}
    for arm in ("zeroshot", "c2lo"):
        no = load(result_files(d, arm, "noalyx"))
        wi = load(result_files(d, arm, "rescored"))
        assert no and wi, f"{arm}: missing files ({len(no)} no-alyx, {len(wi)} rescored)"
        assert len(no) == len(wi), f"{arm}: {len(no)} no-alyx seeds against {len(wi)} rescored"
        for s in no + wi:
            assert "per_cell_user" in s["arms"]["A1"], "a file predates per-cell output; re-score it with the current harness"
        diff = arm_vectors(no, ALL) - arm_vectors(wi, ALL)
        assert diff.shape == (17,), diff.shape
        r = ci(diff, rng)
        out["rows"][f"{arm}_noalyx-withalyx"] = {**r, "seeds": [len(no), len(wi)]}
        out["verdicts"][f"{arm}_noalyx-withalyx"] = verdict(r)
        out["levels"][arm] = {"with_alyx": float(arm_vectors(wi, ALL).mean()), "no_alyx": float(arm_vectors(no, ALL).mean())}
        hl, rest = cells_involving("half_life_alyx"), cells_not_involving("half_life_alyx")
        out["descriptive"][f"{arm}_halflife_cells"] = ci(arm_vectors(no, hl) - arm_vectors(wi, hl), rng)
        out["descriptive"][f"{arm}_other_cells"] = ci(arm_vectors(no, rest) - arm_vectors(wi, rest), rng)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))      # artefact first
    for arm, l in out["levels"].items():
        print(f"  {arm:8s} A1 with alyx {l['with_alyx']:.3f}  without {l['no_alyx']:.3f}")
    for k, r in out["rows"].items():
        print(f"  {k:24s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] -> {out['verdicts'][k]}")
    for k, r in out["descriptive"].items():
        print(f"  (descriptive) {k:22s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""
Read the broad 2 s programme against broad_2s_REGISTERED.md. Committed before any result existed.

Ball-throwing: per arm, per user, each condition averaged over that arm's gated seeds; an arm-vs-arm row uses
only the seeds gated in BOTH arms (seed-paired), then a user bootstrap. Ratios (rho = mean C2 / mean C1) are
bootstrapped as ratios of means over the same resampled users in both arms. Alyx: (seed, user) units, 2 s
and 10 s paired on the same units. Regions and the verdict rule are exposure_breadth_read's (one
implementation); `boot` is across_xr_alignment_aggregate's.

    python docs/acceptance/broad_2s_read.py \
        --bt treatment=docs/acceptance/ballthrowing_cross_day_s1.json [... ARM=file] \
        --alyx docs/acceptance/alyx_cross_day_gpu.json docs/acceptance/broad_2s_alyx.json \
        [--tilt docs/acceptance/ballthrowing_tilt_lookup.json] [--out ...]

Alyx arms read: "treatment" (10 s, the alyx_cross_day result), "treatment_2s", "control_2s".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exposure_breadth_read as R  # noqa: E402
from across_xr_alignment_aggregate import N_BOOT  # noqa: E402

R.ROWS.update({
    "nym_C1": [(-np.inf, 0.0, "FALSIFIER: Nymeria training does not help cross-day identification on an unseen corpus"),
               (0.0, 0.02, "not resolved"),
               (0.02, 0.10, "BAND: daily-life AR-glasses training carries to an unseen activity"),
               (0.10, np.inf, "exceeds: report as such")],
    "nym_persistence": [(-np.inf, -0.03, "Nymeria training makes identification less persistent"),
                        (-0.03, 0.03, "BAND: no change in persistence"),
                        (0.03, np.inf, "Nymeria training makes identification more persistent")],
    "raw_C1": [(-np.inf, 0.0, "raw does not add across days within a headset"),
               (0.0, 0.05, "small"),
               (0.05, np.inf, "BAND: a static cue adds across days within a headset (height/placement, not behaviour)")],
    "raw_headset": [(-np.inf, -0.05, "BAND: a headset change scrambles the static frame raw relies on"),
                    (-0.05, 0.0, "small"),
                    (0.0, np.inf, "FALSIFIER: raw survives a headset change at least as well as dyn")],
    "br_rho": [(-np.inf, -0.05, "FALSIFIER: removing absolute tilt does not reduce the proportional headset cost"),
               (-0.05, 0.10, "not resolved"),
               (0.10, np.inf, "BAND: consistent with tilt carrying headset fit")],
    "tilt_headset": [(-np.inf, -0.05, "BAND: tilt identity is headset-bound"),
                     (-0.05, 0.0, "weak"),
                     (0.0, np.inf, "FALSIFIER: tilt survives a headset change; it cannot account for the headset cost")],
    "alyx_rho_2s": [(-np.inf, 0.75, "BAND: alyx stays costly at 2 s; ball-throwing's smaller day cost is the corpus/task"),
                    (0.75, 0.84, "partly window length"),
                    (0.84, np.inf, "FALSIFIER: the half-size day cost is a window-length effect")],
})
CONDS = ("C0", "C1", "C2")


def load_bt(specs):
    """{arm: {seed: result}} for gated results; refusals listed."""
    arms, refused = defaultdict(dict), []
    for spec in specs:
        arm, path = spec.split("=", 1)
        for r in json.loads(pathlib.Path(path).read_text())["results"]:
            if "refused" in r:
                refused.append((arm, r["seed"], r["refused"]))
                continue
            assert r["seed"] not in arms[arm], f"{arm} seed {r['seed']} given twice"
            arms[arm][r["seed"]] = r
    return arms, refused


def per_user(arm, seeds, users):
    return {k: np.array([np.mean([arm[s][k][u] for s in seeds]) for u in users]) for k in CONDS}


def paired(arms, a, b):
    seeds = sorted(set(arms[a]) & set(arms[b]))
    assert seeds, f"no seed gated in both {a} and {b}"
    users = sorted(arms[a][seeds[0]]["C0"])
    for arm in (a, b):
        for s in seeds:
            assert sorted(arms[arm][s]["C0"]) == users and len(users) == 41, (arm, s)
    return seeds, users, per_user(arms[a], seeds, users), per_user(arms[b], seeds, users)


def boot_ratio_diff(c2a, c1a, c2b=None, c1b=None, rng=None):
    """rho_a - rho_b (or rho_a alone), rho = mean(C2)/mean(C1), users resampled jointly."""
    def stat(i):
        ra = c2a[i].mean() / c1a[i].mean()
        return ra if c2b is None else ra - c2b[i].mean() / c1b[i].mean()
    n = len(c1a)
    full = stat(np.arange(n))
    draws = [stat(rng.integers(0, n, size=n)) for _ in range(N_BOOT)]
    return {"mean": float(full), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def row(out, key, r, extra=None):
    out["rows"][key] = {**r, **(extra or {})}
    out["verdicts"][key] = R.verdict(key, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", nargs="+", required=True, help="ARM=file; arms: treatment, control, raw, br")
    ap.add_argument("--alyx", nargs="*", default=[])
    ap.add_argument("--tilt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(67)
    arms, refused = load_bt(a.bt)
    out = {"refused": refused, "levels": {}, "rows": {}, "verdicts": {}, "preconditions": {}, "descriptive": {}}
    users_ref = None
    for arm in sorted(arms):
        seeds = sorted(arms[arm])
        users = sorted(arms[arm][seeds[0]]["C0"]); users_ref = users_ref or users
        m = per_user(arms[arm], seeds, users)
        out["levels"][arm] = {"seeds": seeds, **{k: R.ci(m[k], rng) for k in CONDS},
                              "best_epochs": {s: arms[arm][s].get("gate", {}).get("best_epoch") for s in seeds}}
    if "control" in arms:
        seeds, _, t, c = paired(arms, "treatment", "control")
        row(out, "nym_C1", R.ci(t["C1"] - c["C1"], rng), {"seeds": seeds})
        row(out, "nym_persistence", R.ci((t["C1"] - t["C0"]) - (c["C1"] - c["C0"]), rng), {"seeds": seeds})
    if "raw" in arms:
        seeds, _, r_, d = paired(arms, "raw", "treatment")
        row(out, "raw_C1", R.ci(r_["C1"] - d["C1"], rng), {"seeds": seeds})
        row(out, "raw_headset", R.ci((r_["C2"] - r_["C1"]) - (d["C2"] - d["C1"]), rng), {"seeds": seeds})
        out["descriptive"]["raw_C0-dyn_C0"] = R.ci(r_["C0"] - d["C0"], rng)
        out["descriptive"]["raw_C2-dyn_C2"] = R.ci(r_["C2"] - d["C2"], rng)
    if "br" in arms:
        seeds, _, b, d = paired(arms, "br", "treatment")
        c1 = R.ci(b["C1"], rng)
        ok = c1["lo"] >= 0.15
        out["preconditions"]["br_C1_lo>=0.15"] = {"C1": c1, "met": ok}
        r = boot_ratio_diff(b["C2"], b["C1"], d["C2"], d["C1"], rng=rng)
        row(out, "br_rho", r, {"seeds": seeds, "rho_br": float(b["C2"].mean() / b["C1"].mean()),
                               "rho_dyn": float(d["C2"].mean() / d["C1"].mean())})
        if not ok:
            out["verdicts"]["br_rho"] = "UNREADABLE: precondition not met (br C1 lower edge < 0.15)"
    if a.tilt:
        t = json.loads(pathlib.Path(a.tilt).read_text())
        assert t["gate"]["passed"], t["gate"]
        users = sorted(t["tilt"]["C1"]); assert users == users_ref
        c1, c2 = (np.array([t["tilt"][k][u] for u in users]) for k in ("C1", "C2"))
        lvl = R.ci(c1, rng)
        out["preconditions"]["tilt_C1_lo>=0.049"] = {"C1": lvl, "C2": R.ci(c2, rng), "met": lvl["lo"] >= 0.049}
        row(out, "tilt_headset", R.ci(c2 - c1, rng))
        if lvl["lo"] < 0.049:
            out["verdicts"]["tilt_headset"] = "UNREADABLE: precondition not met (tilt C1 lower edge < 2x chance)"
    if a.alyx:
        units = defaultdict(dict)
        for f in a.alyx:
            for r in json.loads(pathlib.Path(f).read_text())["results"]:
                if "refused" in r:
                    out["refused"].append(("alyx_" + r["arm"], r.get("seed"), r["refused"]))
                    continue
                if r["arm"] in ("treatment", "treatment_2s", "control_2s"):
                    for u in r["cross_day"]:
                        assert (r["seed"], u) not in units[r["arm"]], (r["arm"], r["seed"], u)
                        units[r["arm"]][(r["seed"], u)] = (r["cross_day"][u], r["same_day"][u])
        if "treatment_2s" in units:
            keys = sorted(units["treatment_2s"])
            x, s = (np.array([units["treatment_2s"][k][i] for k in keys]) for i in (0, 1))
            sd = R.ci(s, rng)
            out["preconditions"]["alyx_2s_same_day_lo>=0.15"] = {"same_day": sd, "cross_day": R.ci(x, rng), "met": sd["lo"] >= 0.15}
            row(out, "alyx_rho_2s", boot_ratio_diff(x, s, rng=rng), {"units": len(keys)})
            if sd["lo"] < 0.15:
                out["verdicts"]["alyx_rho_2s"] = "UNREADABLE: precondition not met (2 s same-day lower edge < 0.15)"
            if "treatment" in units:
                common = [k for k in keys if k in units["treatment"]]
                x10, s10 = (np.array([units["treatment"][k][i] for k in common]) for i in (0, 1))
                x2, s2 = (np.array([units["treatment_2s"][k][i] for k in common]) for i in (0, 1))
                out["descriptive"]["alyx_cost_2s-10s_paired"] = {**R.ci((x2 - s2) - (x10 - s10), rng), "units": len(common),
                                                                  "rho_10s": float(x10.mean() / s10.mean())}
            if "control_2s" in units:
                common = [k for k in keys if k in units["control_2s"]]
                xc = np.array([units["control_2s"][k][0] for k in common]); xt = np.array([units["treatment_2s"][k][0] for k in common])
                out["descriptive"]["alyx_cross_day_treatment_2s-control_2s"] = {**R.ci(xt - xc, rng), "units": len(common)}
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=float))     # artefact first
    for arm, l in out["levels"].items():
        print(f"  {arm:10s} seeds {l['seeds']}  " + "  ".join(f"{k} {l[k]['mean']:.3f}" for k in CONDS))
    for k, r in out["rows"].items():
        print(f"  {k:16s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {out['verdicts'][k]}")
    for k, v in out["preconditions"].items():
        print(f"  precondition {k}: {'met' if v['met'] else 'NOT MET'}")
    for k, r in out["descriptive"].items():
        print(f"  (descriptive) {k:36s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    for s in out["refused"]:
        print("  refused (not read):", s)
    if a.out:
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""
Read the Nymeria activity-diversity arms, with both traps handled in code rather than in a
reader's memory.

TRAP 1 - sweep_id is a mixture. A mode=rescore row inherits the sweep_id of the checkpoint it
scored, so sweep 0840769514 holds five mode=train transfer rows AND five nymeria_rescored rows.
A mean over the id alone returns 0.5685, which is neither quantity, looks entirely plausible,
and would hand the treatment a spurious +0.03 before it ran. Rescoring makes every gated sweep
a mixture, so the id is now NEVER sufficient on its own. `rows()` filters on mode and
experiment and refuses to return a mixed set.

TRAP 2 - patience is a bias correlated with the treatment. Only two of the five control seeds
reached the 120 cap. If Nymeria changes the convergence rate at all, patience truncates one arm
harder than the other and the delta silently carries a budget term. The control's convergence
band is mean best_epoch 98.0 +-18.6; a treatment landing outside it means the delta is not
clean. Checked and printed beside every result, never after.

The control's epochs/patience are recorded as None, so rather than assert they match, they are
DERIVED: three of five seeds have epochs_run == best_epoch + 15 exactly and the other two ran
to 120. That is patience=15 under a 120 cap, from the rows.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHARD = ROOT / "results" / "runs" / "desktop-c.jsonl"
CONTROL_BAND = (98.0, 18.6)   # superseded: the band now comes from each arm's own control
BAND_LOW = 0.005              # the Coordinator's registered band, lower edge
FALSIFIER = 0.005             # registered falsifier: under this says activity diversity did nothing


def load() -> list[dict]:
    return [json.loads(l) for l in SHARD.read_text(encoding="utf-8").splitlines() if l.strip()]


def rows(all_rows, *, experiment=None, sweep_id=None, seeds=None) -> list[dict]:
    """Training rows only. mode=rescore rows inherit their checkpoint's sweep_id, so any
    selection that does not exclude them is a mixture of two different measurements."""
    out = [r for r in all_rows if r.get("mode") == "train"]
    if experiment is not None:
        out = [r for r in out if r.get("experiment") == experiment]
    if sweep_id is not None:
        out = [r for r in out if r.get("sweep_id") == sweep_id]
    if seeds is not None:
        out = [r for r in out if r.get("seed") in seeds]
    modes = {(r.get("mode"), r.get("experiment")) for r in out}
    assert len(modes) <= 1, f"refusing to return a mixed row set: {modes}"

    # DEDUPE BY SEED. Two chain wrappers ran concurrently on 2026-09-08/09 - the first was
    # believed killed but only its harness job had died - and they shared .done markers, so a
    # race let both start the same config before either wrote its marker. The duplicates are
    # bit-identical (same seed, same config, deterministic), so they are harmless as numbers
    # and actively useful as a reproducibility check, but counting one twice inflates n and
    # shrinks the apparent sd. Keep the earliest row per seed and report what was dropped.
    by_seed: dict = {}
    for r in sorted(out, key=lambda r: (r["seed"], r.get("timestamp") or "")):
        by_seed.setdefault(r["seed"], []).append(r)
    dropped = []
    for seed, group in sorted(by_seed.items()):
        for extra in group[1:]:
            same = abs(extra["selected_test_auc"] - group[0]["selected_test_auc"]) < 1e-9
            dropped.append((seed, "identical" if same else "DIFFERENT"))
    if dropped:
        print(f"    [deduped {len(dropped)} duplicate row(s): "
              + ", ".join(f"seed {s} {how}" for s, how in dropped) + "]")
        assert all(how == "identical" for _, how in dropped), \
            "a duplicate run disagreed with its twin - that is not a scheduling artefact"
    return [g[0] for _, g in sorted(by_seed.items())]


def derive_budget(rs) -> str:
    """Recover epochs/patience from what the rows do record, rather than trusting a field
    that is None on the control arm."""
    notes = []
    for r in rs:
        be, er = r.get("best_epoch"), r.get("epochs_run")
        if be is None or er is None:
            notes.append("?")
        elif er == be + 15:
            notes.append("p15")
        elif er == 120:
            notes.append("cap")
        else:
            notes.append(f"UNEXPLAINED({be},{er})")
    return " ".join(notes)


def summarise(label, rs):
    v = [r["selected_test_auc"] for r in rs]
    be = [r["best_epoch"] for r in rs if r.get("best_epoch") is not None]
    print(f"  {label:<26} n={len(v)}  mean {st.mean(v):.4f}"
          f"  sd {st.stdev(v) if len(v) > 1 else float('nan'):.4f}"
          f"  best_epoch mean {st.mean(be) if be else float('nan'):.1f}"
          f"  [{derive_budget(rs)}]")
    return v, be


def paired(label, treat, ctrl):
    """Paired by seed. Two arms are only comparable seed-for-seed."""
    ts = {r["seed"]: r["selected_test_auc"] for r in treat}
    cs = {r["seed"]: r["selected_test_auc"] for r in ctrl}
    common = sorted(set(ts) & set(cs))
    if len(common) < 2:
        print(f"\n  {label}: only {len(common)} paired seed(s) - nothing to test yet")
        return
    d = [ts[s] - cs[s] for s in common]
    mean_d, sd_d = st.mean(d), st.stdev(d)
    n = len(d)
    t = mean_d / (sd_d / n ** 0.5) if sd_d else float("inf")
    crit = stats.t.ppf(0.975, n - 1)
    mdd = crit / n ** 0.5 * sd_d
    print(f"\n  {label}: seeds {common}")
    print(f"    per-seed delta " + "  ".join(f"{x:+.4f}" for x in d))
    print(f"    mean {mean_d:+.4f}   paired sd {sd_d:.4f}   t({n-1})={t:.2f}   won {sum(x>0 for x in d)}/{n}")
    print(f"    MDD at this n and sd: {mdd:.4f}"
          + ("   <- the effect is INSIDE the noise floor; not distinguishable from zero"
             if abs(mean_d) < mdd else "   <- resolved"))

    # "Not resolved" says only that zero is not excluded. It does NOT say the design was
    # uninformative, and reporting it alone can badly understate a decisive negative: an
    # interval can fail to exclude zero while excluding the entire registered band. So the
    # bound is reported beside the test, and the registered thresholds are checked against
    # the interval rather than against the point estimate.
    lo, hi = mean_d - crit * sd_d / n ** 0.5, mean_d + crit * sd_d / n ** 0.5
    print(f"    95% CI [{lo:+.4f}, {hi:+.4f}]")
    for label, thresh in (("registered band lower edge", BAND_LOW), ("falsifier", FALSIFIER)):
        if hi < thresh:
            print(f"      -> {label} {thresh:+.3f} is ABOVE the whole interval: EXCLUDED")
        elif lo > thresh:
            print(f"      -> {label} {thresh:+.3f} is BELOW the whole interval: exceeded")
        else:
            print(f"      -> {label} {thresh:+.3f} lies inside the interval: not settled")


def convergence_check(label, treat, ctrl):
    """The band comes from THIS arm's own control, not a constant. Arm A's control has both
    seeds at the 120 cap (mean 117.0, no early stops), so arm B's 98+-19 is the wrong yardstick
    for it - and a censored control makes the comparison sharper, not looser: if the treatment
    stops on patience while the control never did, that IS a convergence difference."""
    bt = [r["best_epoch"] for r in treat if r.get("best_epoch") is not None]
    bc = [r["best_epoch"] for r in ctrl if r.get("best_epoch") is not None]
    if not bt or not bc:
        return
    m, cm = st.mean(bt), st.mean(bc)
    csd = st.stdev(bc) if len(bc) > 1 else 0.0
    capped_c = sum(1 for r in ctrl if r.get("epochs_run") == 120)
    capped_t = sum(1 for r in treat if r.get("epochs_run") == 120)
    inside = abs(m - cm) <= max(csd, 1e-9)
    print(f"    convergence: treatment best_epoch {m:.1f} against this arm's control {cm:.1f}+-{csd:.1f}"
          f"  (capped {capped_t}/{len(treat)} vs {capped_c}/{len(ctrl)})")
    print(f"      -> {'matched, delta is clean' if inside else 'OUTSIDE the control band - the delta contains a budget term'}")
    if csd == 0.0 and capped_c == len(ctrl) and capped_t < len(treat):
        print("      -> control never stopped early and the treatment did: convergence differs, "
              "so re-run one arm at patience=0 before quoting the delta")


if __name__ == "__main__":
    all_rows = load()
    print("ARM A - Nymeria added to the 4096-identity mix (Nymeria 2.9% of windows)\n")
    a_treat = rows(all_rows, experiment="nymeria_activity")
    a_base_old = [r for r in load() if r.get("sweep_id") in ("f9ca1571b9", "c05d670fff")
                  and r.get("mode") == "train" and r.get("experiment") == "transfer"]
    a_base_new = rows(all_rows, experiment="nymeria_baseline")
    a_ctrl = sorted(a_base_old + a_base_new, key=lambda r: r["seed"])
    if a_ctrl:
        summarise("control (BOXRR+alyx)", a_ctrl)
    if a_treat:
        summarise("treatment (+Nymeria)", a_treat)
        paired("ARM A paired", a_treat, a_ctrl)
        convergence_check("A", a_treat, a_ctrl)

    print("\n\nARM B - activity swapped in at FIXED 419 identities (Nymeria 14.2% of windows)\n")
    b_ctrl = rows(all_rows, experiment="transfer", sweep_id="0840769514")
    b_treat = rows(all_rows, experiment="nymeria_swap")
    if b_ctrl:
        summarise("control (BOXRR343+alyx)", b_ctrl)
    if b_treat:
        summarise("treatment (293+alyx+Nym)", b_treat)
        paired("ARM B paired", b_treat, b_ctrl)
        convergence_check("B", b_treat, b_ctrl)

    print("\n\nPER-CORPUS, for the registered NJIT structure prediction")
    for label, rs in (("A treat", a_treat), ("B treat", b_treat)):
        for r in rs:
            bd = r.get("test_auc_by_dataset") or ""
            if bd:
                d = dict(kv.split("=") for kv in bd.split(";") if "=" in kv)
                njit = float(d.get("NJIT_6DOF_VR_Navigation_Dataset", "nan"))
                others = [float(v) for k, v in d.items() if "NJIT" not in k]
                print(f"  {label} seed {r['seed']}: NJIT {njit:.4f}  others mean {st.mean(others):.4f}")

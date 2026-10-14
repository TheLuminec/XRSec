"""
Results report: one command regenerates every summary table and figure from the committed result files
(Dr. Feng, email 2026-10-12, point 3). Nothing is recomputed from checkpoints; it only reads JSON already on
main. Run it after any new result lands, and attach `docs/results/` to the progress update.

    .venv/bin/python docs/acceptance/results_report.py [--out docs/results] [--push]

Writes:
  summary.md / summary.csv   headline rows, each with metric, N, users-unseen, content-unseen and compliance
                             (split_audit_2026-10-12.md)
  schach_cells.md            ours against Schach et al. per user, all 20 application pairs AND the six pairs
                             no pretraining corpus covers (the content-compliant comparison), both metrics
  schach_cells.png / .svg    per-user comparison figure for the six compliant pairs
  template_verification.md   the deployment decision (template_verification_PROTOCOL.md), every tv_*.json
  tv_per_user_frr.png        per-user false-reject distributions, one strip per arm and condition
  dashboard.json             the feed Avalon Monitor's results page renders (schema xrsec-results/1); --push sends it
Sections whose inputs do not exist yet are reported as missing, never filled in.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import subprocess

import numpy as np

A = pathlib.Path(__file__).resolve().parent
ROOT = A.parents[1]
PUBLISHED = ROOT / "external_sota/schach2026/training-and-evaluation/evaluation/files/slm_model_data/accuracy_values.json"
APPS = {1: "Superhot", 2: "Half-Life: Alyx", 3: "Beat Saber", 4: "Synth Riders", 5: "Social VR"}
UNCOVERED = {1, 4, 5}            # in no pretraining corpus we hold (BOXRR = Beat Saber, alyx = Half-Life)
SERIES = {"theirs": "#2a78d6", "zero_shot": "#eb6834", "c2lo": "#1baf7a"}
N_BOOT, SEED = 10000, 67


def boot(v, rng):
    v = np.asarray(v, float)
    d = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[d].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(m, lo=None, hi=None):
    return f"{m:.3f}" if lo is None else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def jget(path, *keys):
    d = json.loads((A / path).read_text())
    for k in keys:
        d = d[k]
    return d


# ---------------------------------------------------------------- headline rows
# (label, metric, N/chance, value function, users unseen, content unseen, compliance, source)
def _mean_seeds(files, *keys):
    return float(np.mean([jget(f, *keys) for f in files]))


HEADLINE = [
    ("Across-XR zero-shot, all 20 pairs", "rank-1, single window", "17 / 0.059",
     lambda: _mean_seeds([f"across_xr_alignment_seed{s}.json" for s in (1, 2, 3)], "seeds", 0, "arms", "A1", "mean"),
     "not excluded (alyx overlap)", "partly (Beat Saber, Half-Life pretrained)", "users only; depends on alyx (-0.066)", "across_xr_alignment_seed*.json"),
    ("Across-XR zero-shot, 6 uncovered pairs", "rank-1, single window", "17 / 0.059",
     lambda: _mean_seeds([f"across_xr_alignment_seed{s}.json" for s in (1, 2, 3)], "seeds", 0, "arms", "A1", "unseen_activity_cells"),
     "not excluded (alyx overlap)", "yes", "content compliant; alyx overlap not excluded", "across_xr_alignment_seed*.json"),
    ("Across-XR C2-lo (exposure arm)", "rank-1, single window", "17 / 0.059",
     lambda: _mean_seeds([f"across_xr_alignment_c2lo_seed{s}.json" for s in (1, 2, 3)], "seeds", 0, "arms", "A1", "mean"),
     "not excluded (alyx overlap)", "no (all apps seen via users 0-22)", "users only; depends on alyx (-0.052)", "across_xr_alignment_c2lo_seed*.json"),
    ("Across-XR zero-shot, no Who-is-Alyx in training", "rank-1, single window", "17 / 0.059",
     lambda: jget("alyx_axr_overlap_read.json", "levels", "zeroshot", "no_alyx"),
     "yes (no Würzburg corpus in training)", "partly (Beat Saber pretrained)", "users only, overlap-free", "alyx_axr_overlap_read.json"),
    ("Across-XR C2-lo, no Who-is-Alyx in training", "rank-1, single window", "17 / 0.059",
     lambda: jget("alyx_axr_overlap_read.json", "levels", "c2lo", "no_alyx"),
     "yes (no Würzburg corpus in training)", "no (all apps seen via users 0-22)", "users only, overlap-free", "alyx_axr_overlap_read.json"),
    ("Nymeria in-domain, activity hidden", "rank-1 N=17, cell-balanced", "17 / 0.059",
     lambda: jget("nymeria_rank1_read.json", "registered", "treatment_constrained_n17", "mean"),
     "yes", "no (scripts in training)", "users only", "nymeria_rank1_read.json"),
    ("Nymeria in-domain, activity hidden", "verification AUC (constrained)", "pairs / 0.50",
     lambda: float(np.mean([0.6609, 0.6782, 0.6649])), "yes", "no (scripts in training)", "users only",
     "nymeria_in_domain_REGISTERED.md Amendment 17 (GPU)"),
    ("Nymeria leave-script-out", "verification AUC (constrained)", "pairs / 0.50",
     lambda: 0.6145, "yes", "yes (5 scripts held out)", "compliant", "nymeria_lso_REGISTERED.md"),
    ("Ball-throwing, same headset other day", "rank-1, single 2 s throw", "41 / 0.024",
     lambda: jget("ballthrowing_cross_day_read.json", "levels", "C1", "mean"), "yes", "yes", "compliant",
     "ballthrowing_cross_day_read.json"),
    ("Ball-throwing, other headset", "rank-1, single 2 s throw", "41 / 0.024",
     lambda: jget("ballthrowing_cross_day_read.json", "levels", "C2", "mean"), "yes", "yes", "compliant",
     "ballthrowing_cross_day_read.json"),
    ("alyx cross-day", "rank-1, single window", "12-17", lambda: jget("alyx_cross_day_read_gpu.json", "treatment", "cross_day", "mean"),
     "yes", "no (Half-Life in training)", "users only", "alyx_cross_day_read_gpu.json"),
    ("Schach et al. (published), all 20 pairs", "rank-1, their metric", "17 / 0.059",
     lambda: jget("schach_paired.json", "theirs_D1", "level", "mean"), "yes", "partly (trained on all 5 apps)", "theirs", "accuracy_values.json"),
]


def headline(out: pathlib.Path):
    rows = []
    for label, metric, n, fn, uu, cu, comp, src in HEADLINE:
        try:
            val = f"{fn():.3f}"
        except (FileNotFoundError, KeyError, IndexError) as e:
            val = f"MISSING ({type(e).__name__})"
        rows.append([label, metric, n, val, uu, cu, comp, src])
    hdr = ["result", "metric", "N / chance", "value", "users unseen", "content unseen", "compliance", "source"]
    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)
    md = ["# Headline results (regenerated by results_report.py)", "",
          "Compliance follows docs/acceptance/split_audit_2026-10-12.md: users and content both unseen, pretraining included.", "",
          "| " + " | ".join(hdr) + " |", "|" + " --- |" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return rows


# ---------------------------------------------------------------- Schach per user
def schach(out: pathlib.Path):
    rng = np.random.default_rng(SEED)
    d = json.loads((A / "schach_paired.json").read_text())
    cells = [tuple(int(x) for x in c.split("->")) for c in d["cells"]]
    sel = {"all 20 pairs": list(range(len(cells))),
           "6 uncovered pairs": [i for i, (a, b) in enumerate(cells) if a in UNCOVERED and b in UNCOVERED]}
    assert len(sel["6 uncovered pairs"]) == 6
    their_d1 = None
    if PUBLISHED.exists():
        pub = json.loads(PUBLISHED.read_text())
        their_d1 = np.array([next(c for c in pub if c["ref_comment"] == a and c["query_comment"] == b)["accuracies"]["precision_at_1"]
                             for a, b in cells])
        assert np.allclose(their_d1.mean(axis=0), d["theirs_D1"]["per_user"]), "published JSON disagrees with schach_paired.json"
    metrics = {"D2 (our template metric)": ("D2_p1_per_cell_user", np.array(d["theirs_D2"]["p1_per_cell_user"])),
               "D1 (their nearest-reference metric)": ("D1_p1_per_cell_user", their_d1)}
    lines = ["# Ours against Schach et al., per user (users 32-48, N=17)", "",
             "Both arms on one harness per metric (schach_paired.json). The six uncovered pairs are the application "
             "pairs among Superhot, Synth Riders and Social VR, in none of our pretraining corpora: the content-compliant "
             "comparison for our ZERO-SHOT arm only (split_audit_2026-10-12.md). C2-lo trained on Across-XR users 0-22 in "
             "all five applications, and their model on all five, so on these pairs both of those have seen the content and "
             "only zero-shot has not. Descriptive: no band is registered on the six-pair rows.", ""]
    fig_data = {}
    for mname, (key, theirs) in metrics.items():
        if theirs is None:
            lines.append(f"**{mname}: their per-cell values missing ({PUBLISHED} not on disk).**\n")
            continue
        lines += [f"## {mname}", "", "| cells | theirs | zero-shot (ours) | C2-lo (ours) | zero-shot - theirs | C2-lo - theirs | users won (zero-shot / C2-lo) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for sname, idx in sel.items():
            th = theirs[idx].mean(axis=0)
            ours = {arm: np.mean([np.array(d["ours"][arm][f"seed{s}"][key])[idx].mean(axis=0) for s in (1, 2, 3)], axis=0)
                    for arm in ("zero_shot", "c2lo")}
            row = [sname, fmt(*boot(th, rng))] + [fmt(*boot(ours[a], rng)) for a in ("zero_shot", "c2lo")] \
                + [fmt(*boot(ours[a] - th, rng)) for a in ("zero_shot", "c2lo")] \
                + [f"{int((ours['zero_shot'] > th).sum())} / {int((ours['c2lo'] > th).sum())} of 17"]
            lines.append("| " + " | ".join(row) + " |")
            if sname == "6 uncovered pairs" and key.startswith("D2"):
                fig_data = {"theirs": th, **ours}
        lines.append("")
    (out / "schach_cells.md").write_text("\n".join(lines) + "\n")
    if fig_data:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(fig_data["theirs"])
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        y = np.arange(17)
        for name, lab in (("theirs", "Schach et al. (trained on all 5 apps)"), ("zero_shot", "ours, zero-shot"), ("c2lo", "ours, C2-lo")):
            v = fig_data[name][order]
            ax.scatter(v, y, s=28, color=SERIES[name], label=lab, zorder=3)
            ax.axvline(v.mean(), color=SERIES[name], ls="--", lw=1)
        ax.set_yticks(y, [str(32 + i) for i in order]); ax.set_xlabel("rank-1 per user, template metric, 6 uncovered app pairs")
        ax.set_ylabel("test user"); ax.axvline(1 / 17, color="#999", lw=0.8, ls=":"); ax.grid(axis="x", color="#e6e5e1")
        ax.legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout()
        plt.rcParams["svg.hashsalt"] = "xrsec"          # deterministic ids, so a regeneration is not a diff
        fig.savefig(out / "schach_cells.png", dpi=160)
        fig.savefig(out / "schach_cells.svg", dpi=160, metadata={"Date": None})
        plt.close(fig)


# ---------------------------------------------------------------- template verification
def template_verification(out: pathlib.Path):
    files = sorted(A.glob("tv_*_c*.json"))         # cuda and cpu: each is scored on its committed figures' device
    lines = ["# Template verification (template_verification_PROTOCOL.md)", ""]
    if not files:
        (out / "template_verification.md").write_text("\n".join(lines + ["MISSING: no tv_*.json on main yet."]) + "\n")
        return
    lines += ["| corpus | arm | seed | condition | users | EER | FAR / FRR at cross-fitted EER threshold | balanced accuracy | TAR at FAR 1 % (realised FAR) | per-user FRR p10 / p50 / p90 | plumbing |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    strips = {}
    for f in files:
        r = json.loads(f.read_text())
        corpus = f.name.split("_")[1] + (" (cpu)" if f.name.endswith("_cpu.json") else "")
        if "refused" in r:
            lines.append(f"| {corpus} | {r.get('arm')} | {r.get('seed')} | REFUSED: {r['refused']} | | | | | | | |")
            continue
        for cname, c in r["conditions"].items():
            q = c["per_user_frr_quantiles"]; cf = c["crossfit_eer_threshold"]; t1 = c["crossfit_far1pct"]
            lines.append(f"| {corpus} | {r['arm']} | {r['seed']} | {cname} | {c['n_users']} | {c['eer']:.3f} | {cf['far']:.3f} / {cf['frr']:.3f} | "
                         f"{cf['balanced_accuracy']:.3f} | {t1['tar']:.3f} ({t1['realised_far']:.3f}) | {q['0.1']:.2f} / {q['0.5']:.2f} / {q['0.9']:.2f} | "
                         f"{r['plumbing'].get('max_abs_gap', 'n/a')} |")
            strips.setdefault((corpus, r["arm"], cname), []).append(np.array(c["per_user_frr"]))
    (out / "template_verification.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(strips)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(keys) + 1.2))
    for i, k in enumerate(keys):
        arrs = strips[k]
        # seeds averaged per user where the users are the same (axr, ballthrowing); alyx draws different users
        # per seed, so there each (seed, user) is a unit, as in alyx_cross_day_read.py
        v = np.mean(arrs, axis=0) if len({len(x) for x in arrs}) == 1 and k[0] != "alyx" else np.concatenate(arrs)
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), s=12, color="#2a78d6", alpha=0.7)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="#0b0b0b", lw=1.5)
    ax.set_yticks(range(len(keys)), [" / ".join(k) for k in keys], fontsize=7)
    ax.set_xlabel("per-user false-reject rate at the cross-fitted EER threshold (seeds averaged; alyx: seed-user units)")
    ax.grid(axis="x", color="#e6e5e1"); fig.tight_layout()
    fig.savefig(out / "tv_per_user_frr.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------- dashboard feed
# dashboard.json (schema xrsec-results/1) is what Avalon Monitor's results page renders. Aggregate metrics and
# statuses only: no windows, embeddings or checkpoints, and per-user values only by public test-user ID.
def _ci(path, *keys):
    d = jget(path, *keys)
    lo, hi = d.get("ci95") or d.get("ci95_boot") or (d.get("lo"), d.get("hi"))
    return float(d["mean"]), [float(lo), float(hi)] if lo is not None else None


# (id, group, label, metric, n, chance, value -> (mean, ci|None), comparator|None, compliance)
# their model scored on OUR template metric (schach_paired.json theirs_D2), the like-for-like comparator for A1 rows;
# their published 0.180 is on their nearest-reference metric (theirs_D1)
SCHACH = {"label": "Schach et al. on our template metric (head + controllers; published 0.180 is their metric)",
          "value": round(json.loads((A / "schach_paired.json").read_text())["theirs_D2"]["level"]["mean"], 4)}
DASH = [
    ("axr_zeroshot", "Across-XR", "zero-shot, all 20 app pairs", "rank-1, single window", 17, 1 / 17,
     lambda: _ci("across_xr_alignment_aggregate.json", "zero_shot", "arms", "A1"), SCHACH, "with Who-is-Alyx pretrained; depends on it"),
    ("axr_zeroshot_noalyx", "Across-XR", "zero-shot, no Who-is-Alyx in training (overlap-free)", "rank-1, single window", 17, 1 / 17,
     lambda: (jget("alyx_axr_overlap_read.json", "levels", "zeroshot", "no_alyx"), None), SCHACH, "users only, overlap-free"),
    ("axr_zeroshot_uncovered", "Across-XR", "zero-shot, 6 uncovered app pairs", "rank-1, single window", 17, 1 / 17,
     lambda: (_mean_seeds([f"across_xr_alignment_seed{s}.json" for s in (1, 2, 3)], "seeds", 0, "arms", "A1", "unseen_activity_cells"), None),
     None, "content compliant; alyx overlap not excluded"),
    ("axr_c2lo", "Across-XR", "C2-lo (also trained on Across-XR users 0-22)", "rank-1, single window", 17, 1 / 17,
     lambda: _ci("across_xr_alignment_aggregate.json", "C2lo", "arms", "A1"), SCHACH, "with Who-is-Alyx pretrained; depends on it"),
    ("axr_c2lo_noalyx", "Across-XR", "C2-lo, no Who-is-Alyx in training (overlap-free)", "rank-1, single window", 17, 1 / 17,
     lambda: (jget("alyx_axr_overlap_read.json", "levels", "c2lo", "no_alyx"), None), SCHACH, "users only, overlap-free"),
    ("nymeria_rank1", "Nymeria (AR glasses)", "in-domain, activity hidden", "rank-1, cell-balanced", 17, 1 / 17,
     lambda: _ci("nymeria_rank1_read.json", "registered", "treatment_constrained_n17"), None, "users only"),
    ("nymeria_auc", "Nymeria (AR glasses)", "in-domain, activity hidden", "verification AUC (constrained)", None, 0.5,
     lambda: (float(np.mean([0.6609, 0.6782, 0.6649])), None), None, "users only"),
    ("nymeria_lso_auc", "Nymeria (AR glasses)", "leave-script-out", "verification AUC (constrained)", None, 0.5,
     lambda: (0.6145, None), None, "compliant"),
    ("bt_c1", "Ball-throwing", "same headset, other day", "rank-1, single 2 s throw", 41, 1 / 41,
     lambda: _ci("ballthrowing_cross_day_read.json", "levels", "C1"), None, "compliant"),
    ("bt_c2", "Ball-throwing", "other headset", "rank-1, single 2 s throw", 41, 1 / 41,
     lambda: _ci("ballthrowing_cross_day_read.json", "levels", "C2"), None, "compliant"),
    ("alyx_xday", "Who-is-Alyx", "cross-day", "rank-1, single window", 17, 1 / 17,
     lambda: _ci("alyx_cross_day_read_gpu.json", "treatment", "cross_day"), None, "users only"),
]
NOTES = [
    "All figures: users never seen in training, three seeds, gated. Head motion only (comparators use head + both controllers).",
    "Schach et al. ticks are their model on OUR template metric (0.199); their published 0.180 is on their own metric. Zero-shot with alyx (0.234) is a placement, not a beat: Beat Saber and Half-Life content is pretrained.",
    "Across-XR figures DEPEND on Who-is-Alyx in pretraining (zero-shot -0.066, C2-lo -0.052; registered BAND failed). Participant overlap is not excluded until the authors answer: quote the overlap-free no-alyx rows beside the others.",
    "Rack et al. 2023 re-run FAILED its gate (0.850 / 0.566 / 0.174 vs published 0.99 / 0.89 / 0.25): never write 'reproduced'.",
    "Verification AUC, rank-1 and the template accept/reject decision are three different numbers; never compare across them.",
]


def _programmes():
    rows = []
    for line in (A / "INDEX.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("programme",) or set(cells[0]) <= {"-", " "}:
            continue
        title, reg, status, key, _ = cells
        s = status.lower().replace("*", "")
        kind = ("fail" if "fail" in s else "running" if "running" in s else "reference" if "reference" in s
                else "partial" if "first results" in s else "closed")
        rows.append({"id": re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60], "title": title, "status": kind,
                     "status_text": status.replace("*", ""), "key_result": key, "registration": reg.replace("`", "")})
    return rows


def dashboard(out: pathlib.Path):
    import datetime
    headline_rows = []
    for hid, group, label, metric, n, chance, fn, comp, compliance in DASH:
        try:
            v, ci = fn()
        except (FileNotFoundError, KeyError, IndexError):
            v, ci = None, None
        headline_rows.append({"id": hid, "group": group, "result": label, "metric": metric, "n": n, "chance": round(chance, 4),
                              "value": None if v is None else round(v, 4), "ci": None if ci is None else [round(x, 4) for x in ci],
                              "comparator": comp, "compliance": compliance})
    bars, dists = [], []
    tv = {}
    for f in sorted(A.glob("tv_*_c*.json")):
        r = json.loads(f.read_text())
        if "refused" in r or (f.name.startswith("tv_axr") and f.name.endswith("_cuda.json")):
            continue                                  # axr cuda was refused on plumbing; cpu is the scored set
        for cname, c in r["conditions"].items():
            k = (f.name.split("_")[1], r["arm"], cname)
            tv.setdefault(k, []).append(c)
    if tv:
        lab = lambda k: f"{k[0]} {k[1]}: {k[2]}"
        keys = sorted(tv)
        for bid, title, get, better in (("tv_far1", "Template decision: genuine accept rate at 1 % false accept", lambda c: c["crossfit_far1pct"]["tar"], "higher"),
                                ("tv_bacc", "Template decision: balanced accuracy at a cross-fitted threshold", lambda c: c["crossfit_eer_threshold"]["balanced_accuracy"], "higher"),
                                ("tv_eer", "Template decision: equal error rate", lambda c: c["eer"], "lower")):
            bars.append({"id": bid, "title": title, "unit": "rate", "better": better,
                         "rows": [{"label": lab(k), "value": round(float(np.mean([get(c) for c in tv[k]])), 4), "n_seeds": len(tv[k])} for k in keys]})
        dists.append({"id": "tv_frr", "title": "Per-user false-reject rate at the cross-fitted threshold (seeds pooled)",
                      "series": [{"label": lab(k), "values": [round(float(x), 4) for c in tv[k] for x in c["per_user_frr"]]} for k in keys]})
    try:
        o = json.loads((A / "alyx_axr_overlap_read.json").read_text())
        r = lambda x: {"value": round(x["mean"], 4), "ci": [round(x["lo"], 4), round(x["hi"], 4)]}
        bars.append({"id": "alyx_overlap", "title": "Across-XR rank-1 without minus with Who-is-Alyx in training (band ±0.03 predicted; failed)",
                     "unit": "difference", "domain": [-0.15, 0.05], "better": "higher",
                     "band": [-0.03, 0.03], "band_label": "registered prediction ±0.03",
                     "rows": [{"label": "zero-shot, all pairs (registered)", **r(o["rows"]["zeroshot_noalyx-withalyx"])},
                              {"label": "C2-lo, all pairs (registered)", **r(o["rows"]["c2lo_noalyx-withalyx"])},
                              {"label": "zero-shot, Half-Life cells (descriptive)", **r(o["descriptive"]["zeroshot_halflife_cells"])},
                              {"label": "zero-shot, other cells (descriptive)", **r(o["descriptive"]["zeroshot_other_cells"])},
                              {"label": "C2-lo, Half-Life cells (descriptive)", **r(o["descriptive"]["c2lo_halflife_cells"])},
                              {"label": "C2-lo, other cells (descriptive)", **r(o["descriptive"]["c2lo_other_cells"])}]})
    except (FileNotFoundError, KeyError):
        pass
    try:
        d = json.loads((A / "schach_paired.json").read_text())
        th = np.array(d["theirs_D2"]["p1_per_cell_user"]).mean(axis=0)
        ours = {arm: np.mean([np.array(d["ours"][arm][f"seed{s}"]["D2_p1_per_cell_user"]).mean(axis=0) for s in (1, 2, 3)], axis=0)
                for arm in ("zero_shot", "c2lo")}
        dists.append({"id": "schach_per_user", "title": "Across-XR rank-1 per test user (32-48), all 20 pairs, template metric",
                      "series": [{"label": "Schach et al.", "values": [round(float(x), 4) for x in th]},
                                 {"label": "ours, zero-shot", "values": [round(float(x), 4) for x in ours["zero_shot"]]},
                                 {"label": "ours, C2-lo", "values": [round(float(x), 4) for x in ours["c2lo"]]}],
                      "point_labels": [str(32 + i) for i in range(len(th))]})
    except (FileNotFoundError, KeyError):
        pass
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    feed = {"schema": "xrsec-results/1", "project": "xrsec",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "commit": commit, "headline": headline_rows, "programmes": _programmes(), "bars": bars,
            "distributions": dists, "notes": NOTES}
    (out / "dashboard.json").write_text(json.dumps(feed, indent=1) + "\n")
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "results"))
    ap.add_argument("--push", action="store_true", help="push dashboard.json to Avalon Monitor (avm results push xrsec)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = headline(out)
    schach(out)
    template_verification(out)
    dashboard(out)
    if a.push:
        # AVM_CLI overrides the command, e.g. "python3 ~/Projects/System_Monitor/agent/avm.py" until the installed
        # avm has the results subcommand
        import os, shlex
        cli = shlex.split(os.path.expanduser(os.environ.get("AVM_CLI", "avm")))
        subprocess.run(cli + ["results", "push", "xrsec", str(out / "dashboard.json")], check=True)
    for r in rows:
        print(f"  {r[0]:42s} {r[3]:>14s}  {r[6]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

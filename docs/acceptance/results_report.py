"""
Results report: one command regenerates every summary table and figure from the committed result files
(Dr. Feng, email 2026-10-12, point 3). Nothing is recomputed from checkpoints; it only reads JSON already on
main. Run it after any new result lands, and attach `docs/results/` to the progress update.

    .venv/bin/python docs/acceptance/results_report.py [--out docs/results]

Writes:
  summary.md / summary.csv   headline rows, each with metric, N, users-unseen, content-unseen and compliance
                             (split_audit_2026-10-12.md)
  schach_cells.md            ours against Schach et al. per user, all 20 application pairs AND the six pairs
                             no pretraining corpus covers (the content-compliant comparison), both metrics
  schach_cells.png / .svg    per-user comparison figure for the six compliant pairs
  template_verification.md   the deployment decision (template_verification_PROTOCOL.md), every tv_*.json
  tv_per_user_frr.png        per-user false-reject distributions, one strip per arm and condition
Sections whose inputs do not exist yet are reported as missing, never filled in.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

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
     "yes (overlap pending)", "partly (Beat Saber, Half-Life pretrained)", "users only", "across_xr_alignment_seed*.json"),
    ("Across-XR zero-shot, 6 uncovered pairs", "rank-1, single window", "17 / 0.059",
     lambda: _mean_seeds([f"across_xr_alignment_seed{s}.json" for s in (1, 2, 3)], "seeds", 0, "arms", "A1", "unseen_activity_cells"),
     "yes (overlap pending)", "yes", "compliant (pending overlap)", "across_xr_alignment_seed*.json"),
    ("Across-XR C2-lo (exposure arm)", "rank-1, single window", "17 / 0.059",
     lambda: _mean_seeds([f"across_xr_alignment_c2lo_seed{s}.json" for s in (1, 2, 3)], "seeds", 0, "arms", "A1", "mean"),
     "yes (overlap pending)", "no (all apps seen via users 0-22)", "users only", "across_xr_alignment_c2lo_seed*.json"),
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
        for ext in ("png", "svg"):
            fig.savefig(out / f"schach_cells.{ext}", dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------- template verification
def template_verification(out: pathlib.Path):
    files = sorted(A.glob("tv_*_cuda.json"))
    lines = ["# Template verification (template_verification_PROTOCOL.md)", ""]
    if not files:
        (out / "template_verification.md").write_text("\n".join(lines + ["MISSING: no tv_*_cuda.json on main yet."]) + "\n")
        return
    lines += ["| corpus | arm | seed | condition | users | EER | FAR / FRR at cross-fitted EER threshold | balanced accuracy | TAR at FAR 1 % (realised FAR) | per-user FRR p10 / p50 / p90 | plumbing |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    strips = {}
    for f in files:
        r = json.loads(f.read_text())
        corpus = f.name.split("_")[1]
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
        v = np.mean(strips[k], axis=0)            # seeds averaged per user
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), s=12, color="#2a78d6", alpha=0.7)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="#0b0b0b", lw=1.5)
    ax.set_yticks(range(len(keys)), [" / ".join(k) for k in keys], fontsize=7)
    ax.set_xlabel("per-user false-reject rate at the cross-fitted EER threshold (seeds averaged)")
    ax.grid(axis="x", color="#e6e5e1"); fig.tight_layout()
    fig.savefig(out / "tv_per_user_frr.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "results"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = headline(out)
    schach(out)
    template_verification(out)
    for r in rows:
        print(f"  {r[0]:42s} {r[3]:>14s}  {r[6]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

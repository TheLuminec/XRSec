"""
Per-user distribution figure for the Schach comparison (Amendment 8): every value comes from
`schach_paired.json`, nothing is recomputed. Each of the 17 test users (32-48) is one row;
the three dots on a row are the same person scored under one metric by their released model,
our zero-shot arm and our C2-lo arm (seeds averaged inside the user). Left panel: their
nearest-reference metric (their calculator). Right panel: our template metric. Users are
ordered by their model's value under their metric in both panels, so a row is the same
person on both sides. Dashed verticals are the population means.

Writes schach_per_user.svg / .png and schach_per_user.csv beside the JSON.

    .venv-eval/bin/python docs/acceptance/schach_per_user_figure.py
"""
from __future__ import annotations

import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
# Reference categorical palette, slots 1-3 (documented as validated all-pairs in light mode).
SERIES = {"their model": "#2a78d6", "zero-shot (ours)": "#eb6834", "C2-lo (ours)": "#1baf7a"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def main() -> int:
    d = json.load(open(HERE / "schach_paired.json"))
    users = np.arange(32, 49)
    panels = {
        "their metric: nearest reference window, 15 s / 10 s": {
            "their model": np.array(d["theirs_D1"]["per_user"]),
            "zero-shot (ours)": np.array(d["ours"]["zero_shot"]["D1"]["per_user"]),
            "C2-lo (ours)": np.array(d["ours"]["c2lo"]["D1"]["per_user"]),
        },
        "our metric: mean template, single probe window": {
            "their model": np.array(d["theirs_D2"]["per_user"]),
            "zero-shot (ours)": np.array(d["ours"]["zero_shot"]["D2"]["per_user"]),
            "C2-lo (ours)": np.array(d["ours"]["c2lo"]["D2"]["per_user"]),
        },
    }
    order = np.argsort(panels[list(panels)[0]]["their model"])   # one ordering for both panels
    rows = [["user"] + [f"{p.split(':')[0]} / {s}" for p in panels for s in SERIES]]
    for i in order[::-1]:
        rows.append([int(users[i])] + [f"{panels[p][s][i]:.4f}" for p in panels for s in SERIES])
    with open(HERE / "schach_per_user.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True, facecolor=SURFACE)
    y = np.arange(len(users))
    for ax, (title, series) in zip(axes, panels.items()):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        # connector per user from the lowest to the highest of the three values
        lo = np.min([series[s][order] for s in SERIES], axis=0)
        hi = np.max([series[s][order] for s in SERIES], axis=0)
        ax.hlines(y, lo, hi, color=GRID, linewidth=2, zorder=1)
        for k, (name, color) in enumerate(SERIES.items()):
            v = series[name][order]
            ax.scatter(v, y, s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=name)
            ax.axvline(v.mean(), color=color, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
            # mean labels staggered by series so neighbouring means never collide
            ax.text(v.mean(), len(users) - 0.4 + 0.75 * k, f"mean {v.mean():.3f}", color=TEXT2, fontsize=8,
                    ha="center", va="bottom", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.axvline(1 / 17, color=TEXT2, linewidth=0.8, linestyle=":", zorder=2)
        ax.text(1 / 17, -0.9, "chance", color=TEXT2, fontsize=8, ha="center", va="top")
        ax.set_xlim(0, 0.75)
        ax.set_ylim(-1.2, len(users) + 2.0)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left", pad=14)
        ax.set_xlabel("cross-application rank-1 at N = 17, mean over 20 ordered cells", fontsize=9, color=TEXT2)
        ax.tick_params(colors=TEXT2, labelsize=8.5, length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"user {int(u)}" for u in users[order]], color=TEXT2)
    axes[0].set_ylabel("Schach et al.'s 17 test users, ordered by their model under their metric", fontsize=9, color=TEXT2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005), labelcolor=TEXT)
    fig.suptitle("The same 17 people under two metrics: their released similarity model against our head-only arms",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"schach_per_user.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote", HERE / "schach_per_user.svg", "and .png / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

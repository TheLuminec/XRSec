"""
Stage 2 of the Schach paired comparison (Amendment 8): one harness per direction, both arms.

D1  our embeddings (schach_embed_ours.py .npz) scored by THEIR MotionAccuracyCalculator,
    imported verbatim from their repository: query = every 10 s stride-5 window of application
    B, reference = application-A windows subsampled per user to one per 25 s ([::5] of the
    stride-5 list), CosineSimilarity nearest neighbour, precision_at_1 per class. Their own
    per-user values come from accuracy_values.json, which schach_release_gate.py has reproduced
    from their embeddings with the same calculator.
D2  THEIR embeddings (schach_release_gate.npz) scored by OUR harness (across_xr_alignment.py's
    centroids / rank1_per_user): gallery = renormalised mean of all application-A windows,
    probe = every application-B window; our A1 re-scored by the same functions in this run and
    checked against the certificate's per-seed means.

Per-user quantity = mean over the 20 ordered cross cells; seeds averaged inside each user;
paired difference ours - theirs; cluster bootstrap over the 17 users (10,000) with the
Student-t interval beside it; outcome BEAT / LOSS / UNRESOLVED by where the 95% interval falls.

    .venv-eval/bin/python docs/acceptance/schach_paired.py --release <training-and-evaluation dir> \
        --theirs docs/acceptance/schach_release_gate.npz --ours-dir <dir with ours_*.npz> \
        --out docs/acceptance/schach_paired.json [--ours-only]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import across_xr_alignment as axa  # noqa: E402

TEST_USERS = list(range(32, 49))
APPS = [1, 2, 3, 4, 5]
APP_NAME = {1: "superhot_vr", 2: "half_life_alyx", 3: "beat_saber", 4: "synth_riders", 5: "social_vr"}
CROSS = [(a, b) for a in APPS for b in APPS if a != b]
N_BOOT = 10000
# Their windows: 450 frames at 30 fps, frame step 5 -> one window per 1/6 s.
THEIR_FPS, THEIR_WINDOW, THEIR_STEP = 30, 450, 5
# Ours: 10 s at 20 Hz, stride 5 s (100 frames).
OUR_FPS, OUR_WINDOW, OUR_STEP = 20, 200, 100
OUR_REF_EVERY = 5     # one reference per 25 s, matching their [::150] of 1/6 s windows
SEQ_SECONDS = 600.0


def make_calculator(release: pathlib.Path, fps: int, window: int, step: int, seq_step_seconds: float):
    sys.path.insert(0, str(release))
    from src.log_metrics.accuracy_calculator import MotionAccuracyCalculator
    from pytorch_metric_learning.utils.inference import CustomKNN
    from pytorch_metric_learning.losses import ProxyAnchorLoss
    import torch
    torch.set_num_threads(4)
    loss = ProxyAnchorLoss(num_classes=23, embedding_size=8)
    return MotionAccuracyCalculator(sequence_lengths_minutes=[10], sliding_window_step_size_seconds=seq_step_seconds,
                                    k="max_bin_count", device=torch.device("cpu"),
                                    exclude=["mean_average_precision", "mean_average_precision_at_r",
                                             "mean_reciprocal_rank", "AMI", "NMI", "r_precision"],
                                    knn_func=CustomKNN(loss.distance), query_fps=fps, query_window_size=window,
                                    query_frame_step_size=step, test_mode=True, return_per_class=True)


def interval(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    mean, lo, hi = axa.bootstrap_mean(values, N_BOOT, rng)
    n = len(values)
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return {"mean": mean, "ci95_boot": [lo, hi], "ci95_t": [mean - half, mean + half],
            "sd_users": float(values.std(ddof=1)), "n_users": int(n)}


def outcome(iv: dict) -> str:
    lo, hi = iv["ci95_boot"]
    return "BEAT" if lo > 0 else ("LOSS" if hi < 0 else "UNRESOLVED")


# ---------------------------------------------------------------- D1: their calculator on ours

def ours_by_their_calculator(npz, calc) -> dict:
    emb, uid, app, start = npz["embeddings"], npz["user_id"], npz["app"], npz["start"]
    p1 = np.zeros((len(CROSS), 17))
    s10 = np.full((len(CROSS), 17), np.nan)
    nref = []
    for ci, (a, b) in enumerate(CROSS):
        q_rows, q_lab, r_rows, r_lab = [], [], [], []
        for li, u in enumerate(TEST_USERS):
            rb = np.where((uid == u) & (app == b))[0]
            rb = rb[np.argsort(start[rb], kind="stable")]
            ra = np.where((uid == u) & (app == a))[0]
            ra = ra[np.argsort(start[ra], kind="stable")][::OUR_REF_EVERY]
            q_rows.append(rb); q_lab.append(np.full(len(rb), li)); r_rows.append(ra); r_lab.append(np.full(len(ra), li))
        q_rows, q_lab, r_rows, r_lab = map(np.concatenate, (q_rows, q_lab, r_rows, r_lab))
        acc = calc.get_accuracy(emb[q_rows], q_lab, emb[r_rows], r_lab)
        p1[ci] = acc["precision_at_1"]
        seq = acc["sequence_top_1_accuracy_list_10_mins"]
        if len(seq) == 17:
            s10[ci] = seq
        nref.append(int(len(r_lab)))
    return {"p1": p1, "seq10": s10, "n_ref": nref}


# ---------------------------------------------------------------- D2: our harness on either side

def template_rank1(emb, user_of_row, app_of_row, order_key, users) -> dict:
    """Per-cell per-user rank-1 (mean template of all app-A windows, every app-B window as
    probe) and the ten-minute majority vote over the first SEQ_SECONDS of each user's probe
    stream; order_key gives each row's time within its (user, app) stream."""
    p1 = np.zeros((len(CROSS), len(users)))
    s10 = np.zeros((len(CROSS), len(users)))
    for ci, (a, b) in enumerate(CROSS):
        g_rows = [np.where((user_of_row == u) & (app_of_row == a))[0] for u in users]
        p_rows = [np.where((user_of_row == u) & (app_of_row == b))[0] for u in users]
        gallery = axa.centroids(emb, g_rows)
        probe_rows = np.concatenate(p_rows)
        probe_user = np.concatenate([np.full(len(r), i) for i, r in enumerate(p_rows)])
        probes = emb[probe_rows]
        p1[ci] = axa.rank1_per_user(gallery, probes, probe_user, len(users))
        pred = axa.rank1_predictions(gallery, probes)
        t = order_key[probe_rows]
        for i in range(len(users)):
            mask = probe_user == i
            t0 = t[mask].min()
            window = mask & (t < t0 + SEQ_SECONDS)
            votes = np.bincount(pred[window], minlength=len(users))
            top = np.flatnonzero(votes == votes.max())
            s10[ci, i] = 1.0 if (len(top) == 1 and top[0] == i) else (1.0 / len(top) if i in top else 0.0)
    return {"p1": p1, "seq10": s10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--theirs", required=True, help="schach_release_gate.npz")
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours-only", action="store_true", help="development: skip everything needing their embeddings")
    args = ap.parse_args()
    release = pathlib.Path(args.release)
    rng = np.random.default_rng(67)
    out = {"registered": "docs/acceptance/across_xr_alignment_REGISTERED.md (Amendment 8)", "cells": [f"{a}->{b}" for a, b in CROSS]}

    # Their per-user values, from the JSON (reproduced by schach_release_gate.py).
    published = json.load(open(release / "evaluation" / "files" / "slm_model_data" / "accuracy_values.json"))
    their_p1 = np.zeros((len(CROSS), 17)); their_s10 = np.zeros((len(CROSS), 17))
    for ci, (a, b) in enumerate(CROSS):
        cell = next(c for c in published if c["ref_comment"] == a and c["query_comment"] == b)
        # A skipped class would shorten a list silently and shift every index after it.
        assert len(cell["accuracies"]["precision_at_1"]) == 17 and len(cell["accuracies"]["sequence_top_1_accuracy_list_10_mins"]) == 17, (a, b)
        their_p1[ci] = cell["accuracies"]["precision_at_1"]
        their_s10[ci] = cell["accuracies"]["sequence_top_1_accuracy_list_10_mins"]
    theirs_d1 = their_p1.mean(axis=0)
    out["theirs_D1"] = {"per_user": theirs_d1.tolist(), "level": interval(theirs_d1, rng),
                        "seq10_per_user": their_s10.mean(axis=0).tolist(), "seq10_level": interval(their_s10.mean(axis=0), rng)}
    print(f"theirs, their metric: {theirs_d1.mean():.4f}; 10-min {their_s10.mean():.4f}", flush=True)

    # Our embeddings: D1 through their calculator, D2 through our harness.
    # Their sequence code applies round(seq_step_seconds * fps) as a stride over the WINDOW axis
    # (30 windows of 1/6 s = 5 s for them); 1/20 s * 20 = 1 of our stride-5 windows = the same 5 s.
    # window_size = round((600*20 - 200)/100) = 118 windows = exactly 600 s of probe stream.
    calc_ours = make_calculator(release, OUR_FPS, OUR_WINDOW, OUR_STEP, seq_step_seconds=1.0 / OUR_FPS)
    # The translation asserted, not assumed (their formulas, our grid): 118 windows = 600 s, step 1 window.
    assert int(np.round(((10 * 60 * OUR_FPS) - OUR_WINDOW) / OUR_STEP)) == 118
    assert int(np.round((1.0 / OUR_FPS) * OUR_FPS)) == 1
    assert int(np.round(((10 * 60 * THEIR_FPS) - THEIR_WINDOW) / THEIR_STEP)) == 3510 and int(np.round(1 * THEIR_FPS)) == 30
    cert = {}
    for name in ("seed1", "seed2", "seed3", "c2lo_seed1", "c2lo_seed2", "c2lo_seed3"):
        p = HERE / f"across_xr_alignment_{name}.json"
        if p.exists():
            cert[name] = json.load(open(p))["seeds"][0]["arms"]["A1"]
    arms = {}
    REGISTERED = ("zero_shot", "c2lo")          # Amendment 8; raw arms report levels only (Amendment 6: headline stays on dyn)
    found = sorted({f.name[len("ours_"):f.name.rindex("_seed")] for f in pathlib.Path(args.ours_dir).glob("ours_*_seed*.npz")})
    assert set(REGISTERED) <= set(found), found
    for arm in REGISTERED + tuple(a for a in found if a not in REGISTERED):
        files = sorted(pathlib.Path(args.ours_dir).glob(f"ours_{arm}_seed*.npz"))
        assert len(files) == (3 if arm in REGISTERED else len(files)), files
        d1_runs, d2_runs, seeds = [], [], []
        for f in files:
            npz = np.load(f)
            seed = int(npz["seed"])
            t0 = time.time()
            d1 = ours_by_their_calculator(npz, calc_ours)
            uid, app, start = npz["user_id"], npz["app"], npz["start"]
            d2 = template_rank1(npz["embeddings"], uid, app, start.astype(float), TEST_USERS)
            cname = f"seed{seed}" if arm == "zero_shot" else f"{arm}_seed{seed}"
            check = None
            if cname in cert:
                check = {"certificate_A1": cert[cname]["mean"], "rescored_A1": float(d2["p1"].mean()),
                         "max_abs_diff_per_user": float(np.max(np.abs(np.array(cert[cname]["per_user"]) - d2["p1"].mean(axis=0))))}
            d1_runs.append(d1); d2_runs.append(d2); seeds.append(seed)
            print(f"  {arm} seed {seed}: their-metric p@1 {d1['p1'].mean():.4f} (10-min {np.nanmean(d1['seq10']):.4f}); "
                  f"our-metric A1 {d2['p1'].mean():.4f} (10-min {d2['seq10'].mean():.4f}); "
                  f"certificate check {check}  ({time.time() - t0:.0f}s)", flush=True)
            if check is not None:
                assert check["max_abs_diff_per_user"] <= 2e-3, check
            arms.setdefault(arm, {})[f"seed{seed}"] = {
                "checkpoint": str(npz["checkpoint"]), "run_id": str(npz["run_id"]),
                "D1_p1_per_cell_user": d1["p1"].tolist(), "D1_seq10_per_cell_user": d1["seq10"].tolist(), "D1_n_ref": d1["n_ref"],
                "D2_p1_per_cell_user": d2["p1"].tolist(), "D2_seq10_per_cell_user": d2["seq10"].tolist(), "certificate_check": check}
        d1_user = np.mean([r["p1"].mean(axis=0) for r in d1_runs], axis=0)       # seeds averaged inside users
        d2_user = np.mean([r["p1"].mean(axis=0) for r in d2_runs], axis=0)
        d1_cell = np.mean([r["p1"] for r in d1_runs], axis=0)                     # 20 x 17
        d2_cell = np.mean([r["p1"] for r in d2_runs], axis=0)
        d1_seq_user = np.mean([np.nanmean(r["seq10"], axis=0) for r in d1_runs], axis=0)
        d2_seq_user = np.mean([r["seq10"].mean(axis=0) for r in d2_runs], axis=0)
        arms[arm]["seeds_scored"] = seeds
        arms[arm]["D1"] = {"per_user": d1_user.tolist(), "level": interval(d1_user, rng),
                           "per_seed_mean": [float(r["p1"].mean()) for r in d1_runs],
                           "seq10_per_seed_mean": [float(np.nanmean(r["seq10"])) for r in d1_runs],
                           "seq10_level": interval(d1_seq_user, rng),
                           "per_cell_mean": d1_cell.mean(axis=1).tolist()}
        arms[arm]["D2"] = {"per_user": d2_user.tolist(), "level": interval(d2_user, rng),
                           "per_seed_mean": [float(r["p1"].mean()) for r in d2_runs],
                           "seq10_per_seed_mean": [float(r["seq10"].mean()) for r in d2_runs],
                           "seq10_level": interval(d2_seq_user, rng),
                           # per ordered cell over the seeds scored, bootstrapped over users (G16)
                           "per_cell": {f"{APP_NAME[a]}->{APP_NAME[b]}": interval(d2_cell[ci], rng) for ci, (a, b) in enumerate(CROSS)}}
        print(f"{arm}: our metric A1 {d2_user.mean():.4f} {arms[arm]['D2']['level']['ci95_boot']}, 10-min {d2_seq_user.mean():.4f} "
              f"{arms[arm]['D2']['seq10_level']['ci95_boot']}; their metric {d1_user.mean():.4f}, 10-min {d1_seq_user.mean():.4f} "
              f"{arms[arm]['D1']['seq10_level']['ci95_boot']}", flush=True)
        if arm not in REGISTERED:
            continue
        diff = d1_user - theirs_d1
        cell_diff = d1_cell - their_p1
        seq_diff = d1_seq_user - their_s10.mean(axis=0)
        arms[arm]["vs_theirs_D1"] = {"paired": interval(diff, rng), "outcome": outcome(interval(diff, rng)),
                                     "per_user_diff": diff.tolist(), "won_users": int((diff > 0).sum()),
                                     "per_cell": {f"{APP_NAME[a]}->{APP_NAME[b]}": interval(cell_diff[ci], rng) for ci, (a, b) in enumerate(CROSS)},
                                     "seq10_paired": interval(seq_diff, rng), "seq10_outcome": outcome(interval(seq_diff, rng))}
        print(f"{arm}: vs theirs (their metric) {diff.mean():+.4f} boot {arms[arm]['vs_theirs_D1']['paired']['ci95_boot']} "
              f"-> {arms[arm]['vs_theirs_D1']['outcome']}; 10-min {seq_diff.mean():+.4f} "
              f"{arms[arm]['vs_theirs_D1']['seq10_paired']['ci95_boot']} -> {arms[arm]['vs_theirs_D1']['seq10_outcome']}", flush=True)
    out["ours"] = arms
    if args.ours_only:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        return 0

    # Their embeddings through our harness (D2).
    th = np.load(args.theirs)
    emb_t, lab_t, com_t = th["embeddings"], th["label"].astype(int), th["comments"].astype(int)
    # time within each (label, app) stream: rows are in file order, i.e. chronological, one per 1/6 s
    order = np.zeros(len(lab_t))
    for li in range(17):
        for c in APPS:
            r = np.where((lab_t == li) & (com_t == c))[0]
            order[r] = np.arange(len(r)) * THEIR_STEP / THEIR_FPS
    t0 = time.time()
    d2t = template_rank1(emb_t, lab_t, com_t, order, list(range(17)))
    theirs_d2 = d2t["p1"].mean(axis=0)
    out["theirs_D2"] = {"per_user": theirs_d2.tolist(), "level": interval(theirs_d2, rng),
                        "per_cell_mean": d2t["p1"].mean(axis=1).tolist(), "seq10_mean": float(d2t["seq10"].mean()),
                        "p1_per_cell_user": d2t["p1"].tolist()}
    print(f"theirs, our metric: {theirs_d2.mean():.4f}; 10-min {d2t['seq10'].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    out["theirs_D2"]["seq10_level"] = interval(d2t["seq10"].mean(axis=0), rng)
    for arm in REGISTERED:
        diff = np.array(arms[arm]["D2"]["per_user"]) - theirs_d2
        iv = interval(diff, rng)
        seq_diff = np.mean([np.array(v["D2_seq10_per_cell_user"]).mean(axis=0) for k, v in arms[arm].items() if k.startswith("seed") and isinstance(v, dict)], axis=0) - d2t["seq10"].mean(axis=0)
        arms[arm]["vs_theirs_D2"] = {"paired": iv, "outcome": outcome(iv), "per_user_diff": diff.tolist(),
                                     "won_users": int((diff > 0).sum()),
                                     "seq10_paired": interval(seq_diff, rng), "seq10_outcome": outcome(interval(seq_diff, rng))}
        print(f"{arm}: D2 level {arms[arm]['D2']['level']['mean']:.4f}, vs theirs {diff.mean():+.4f} "
              f"boot {iv['ci95_boot']} -> {outcome(iv)}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Leak check for leave-one-application-out checkpoints, independent of the node that trained them.

A checkpoint's normaliser stores per-dataset channel statistics fitted on exactly its TRAINING
windows (model/normalization.py: fit on the training index, after encoding). For a LOAO checkpoint
those are Across-XR users 0-22 of CrossApplicationXR_LOAO_<X>. So recompute them, through the
pipeline's own SampleDataset / SampleIndex / ChannelNormalizer, under six hypotheses - each of the
five applications removed, and none removed (the full corpus) - and ask which the stored statistics
match. A clean checkpoint matches "X removed" to float precision and nothing else; a leak matches
"none removed" or a different application.

    python docs/acceptance/loao_leak_check.py --checkpoints X=path [X=path ...] [--out ...]

Positive control before use: the P3 checkpoints (hold-out established) must each match their own X.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
from across_xr_alignment import quiet  # noqa: E402

GAMES = ("superhot_vr", "half_life_alyx", "beat_saber", "synth_riders", "social_vr")
TRAIN_USERS = set(range(0, 23))          # Schach's train split; 23-31 validation, 32-48 test
MATCH_TOL = 1e-5                         # float32 statistics of a float64 accumulation


def hypothesis_stats(corpus_dir: pathlib.Path, es: dict, channels: str):
    from dataset import SampleDataset, SampleIndex
    from normalization import ChannelNormalizer
    ds = quiet(SampleDataset, str(corpus_dir / "users"), sample_time=int(es["sample_time"]),
               sample_rate=int(es["sample_rate"]), channels=channels,
               resample=es.get("resample", "nearest"), window_stride=es.get("window_stride"))
    index = SampleIndex(ds, encoding=es.get("encoding", "raw"))
    rows = [r for d, r in zip(ds.user_dirs, index.user_sample_indices) if int(pathlib.Path(d).name) in TRAIN_USERS]
    assert len(rows) == 23, f"{corpus_dir}: {len(rows)} training users"
    idx = torch.cat(rows)
    return ChannelNormalizer._channel_statistics(index.samples, idx), int(idx.numel())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="X=path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cache, results = {}, {}
    for item in args.checkpoints:
        x, path = item.split("=", 1)
        assert x in GAMES, x
        ck = torch.load(path, map_location="cpu", weights_only=False)
        es = ck["eval_split"]
        stored = ck["normalizer"]["statistics"].get(f"CrossApplicationXR_LOAO_{x}")
        assert stored is not None, f"{path} holds no statistics under CrossApplicationXR_LOAO_{x}"
        s_mean, s_std = (torch.tensor(stored[k], dtype=torch.float32) for k in ("mean", "std"))  # state_dict layout
        key = (es["sample_time"], es["sample_rate"], es.get("window_stride"), es.get("encoding"), ck.get("channels", "full"))
        rec = {}
        for hyp in ("none_removed",) + GAMES:
            corpus = ROOT / "processed_datasets" / ("CrossApplicationXR_Dataset" if hyp == "none_removed"
                                                    else f"CrossApplicationXR_LOAO_{hyp}")
            if (hyp, key) not in cache:
                cache[(hyp, key)] = hypothesis_stats(corpus, es, ck.get("channels", "full"))
            (m, s), n = cache[(hyp, key)]
            gap = max(float((m - s_mean).abs().max()), float((s - s_std).abs().max()))
            rec[hyp] = {"max_abs_gap": gap, "windows": n}
        matches = [h for h, r in rec.items() if r["max_abs_gap"] < MATCH_TOL]
        verdict = ("CLEAN: matches only its own hold-out" if matches == [x] else
                   f"NOT CLEAN or ambiguous: matches {matches}")
        results[x] = {"checkpoint": path, "hypotheses": rec, "matches": matches, "verdict": verdict}
        print(f"{x:15s} " + "  ".join(f"{h}={r['max_abs_gap']:.1e}" for h, r in rec.items()) + f"  -> {verdict}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if all(r["matches"] == [x] for x, r in results.items()) else 1


if __name__ == "__main__":
    sys.exit(main())

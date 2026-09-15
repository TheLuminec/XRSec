"""
Schach et al. release: safety scan, whitelisted load, and the two known-answer reconstructions
that tie their `embeddings.pkl` to their data and to their `accuracy_values.json`
(Amendment 8, before any pairing).

1. `pickletools` scan of the pickle: every GLOBAL / STACK_GLOBAL it references is recorded.
2. Load through an Unpickler whose find_class admits numpy, pandas, collections and builtins
   only - anything else raises before it is imported.
3. Per-(label, application) window counts must equal len(range(0, rows - 450, 5)) from the
   released test CSVs' row counts (`test_rows_per_user_comment.csv`, awk'd over every row on
   AVALON), pinning label i to user 32+i without the sort assumption.
4. Their own MotionAccuracyCalculator, imported verbatim from their repository and built with
   the arguments of their `_compute_accuracy_task`, run on their embeddings for all 35 cells,
   must reproduce every per-class `precision_at_1` list (and the ten-minute sequence lists).

    .venv-eval/bin/python docs/acceptance/schach_release_gate.py --release <training-and-evaluation dir> \
        --rows <test_rows_per_user_comment.csv> --out docs/acceptance/schach_release_gate.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import pathlib
import pickle
import pickletools
import sys
import time

import numpy as np

ALLOWED_MODULE_PREFIXES = ("numpy", "pandas", "collections", "builtins", "_codecs")
WINDOW, STEP, REF_EVERY = 450, 5, 150


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_globals(data: bytes) -> dict[str, int]:
    globs: collections.Counter = collections.Counter()
    last: list[str] = []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING", "SHORT_BINSTRING", "BINSTRING"):
            last.append(arg)
            last = last[-2:]
        elif op.name == "STACK_GLOBAL":
            globs[" ".join(last)] += 1
        elif op.name == "GLOBAL":
            globs[arg] += 1
        elif op.name in ("INST", "OBJ"):
            globs[f"{op.name} {arg}"] += 1
    return dict(globs)


class WhitelistUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if not module.startswith(ALLOWED_MODULE_PREFIXES):
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        return super().find_class(module, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-sequence", action="store_true")
    args = ap.parse_args()
    rel = pathlib.Path(args.release)
    pkl = rel / "evaluation" / "files" / "slm_model_data" / "embeddings.pkl"
    jsn = rel / "evaluation" / "files" / "slm_model_data" / "accuracy_values.json"
    ckpt = rel / "evaluation" / "models" / "slm_model" / "max_precision_at_1.ckpt"
    out = {"files": {str(p.relative_to(rel)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (pkl, jsn, ckpt)}}
    print(json.dumps(out["files"], indent=1), flush=True)

    t0 = time.time()
    data = pkl.read_bytes()
    out["pickle_globals"] = scan_globals(data)
    print("pickle GLOBALs:", json.dumps(out["pickle_globals"], indent=1), f"({time.time() - t0:.0f}s)", flush=True)
    bad = [g for g in out["pickle_globals"] if not g.split(" ")[0].startswith(ALLOWED_MODULE_PREFIXES)]
    assert not bad, f"pickle references modules outside the whitelist: {bad}"
    frame = WhitelistUnpickler(io.BytesIO(data)).load()
    del data
    import pandas as pd
    assert isinstance(frame, pd.DataFrame), type(frame)
    print("loaded", frame.shape, list(frame.columns), flush=True)
    labels = frame["label"].to_numpy().astype(int)
    comments = frame["comments"].to_numpy().astype(int)
    emb = np.stack(frame["embedding"].to_numpy()).astype(np.float32)
    assert emb.shape[1] == 480, emb.shape
    assert np.array_equal(frame["embedding_index"].to_numpy(), np.arange(len(frame)))
    out["shape"] = list(emb.shape)

    # 3. counts reconstruction
    expected = {}
    for u, c, n, z in csv.reader(open(args.rows)):
        assert int(z) == 1
        expected[(int(u), int(c))] = len(range(0, int(n) - WINDOW, STEP))
    counts = collections.Counter(zip(labels.tolist(), comments.tolist()))
    assert sorted(set(labels.tolist())) == list(range(17)), sorted(set(labels.tolist()))
    assert sorted(set(comments.tolist())) == [1, 2, 3, 4, 5]
    mism = []
    for (u, c), n in expected.items():
        if counts[(u - 32, c)] != n:
            mism.append({"user": u, "app": c, "expected": n, "in_pickle": counts[(u - 32, c)]})
    vec = {u: tuple(expected[(u, c)] for c in range(1, 6)) for u in range(32, 49)}
    assert len(set(vec.values())) == 17, "two users share a count vector; the pinning would be ambiguous"
    out["counts"] = {"total_expected": sum(expected.values()), "total_in_pickle": int(len(labels)),
                     "mismatches": mism, "count_vectors_distinct": True,
                     "per_user_expected": {str(u): list(v) for u, v in vec.items()}}
    print("counts:", {k: v for k, v in out["counts"].items() if k != "per_user_expected"}, flush=True)
    assert not mism and len(labels) == sum(expected.values())

    # 4. their calculator on their embeddings, every cell of the JSON
    sys.path.insert(0, str(rel))
    from src.log_metrics.accuracy_calculator import MotionAccuracyCalculator  # noqa: E402
    from pytorch_metric_learning.utils.inference import CustomKNN  # noqa: E402
    from pytorch_metric_learning.losses import ProxyAnchorLoss  # noqa: E402
    import torch
    torch.set_num_threads(4)
    loss = ProxyAnchorLoss(num_classes=23, embedding_size=480)
    seq = [10] if not args.skip_sequence else []
    calc = MotionAccuracyCalculator(sequence_lengths_minutes=seq, sliding_window_step_size_seconds=1,
                                    k="max_bin_count", device=torch.device("cpu"),
                                    exclude=["mean_average_precision", "mean_average_precision_at_r",
                                             "mean_reciprocal_rank", "AMI", "NMI", "r_precision"],
                                    knn_func=CustomKNN(loss.distance), query_fps=30, query_window_size=WINDOW,
                                    query_frame_step_size=STEP, test_mode=True, return_per_class=True)
    published = json.load(open(jsn))
    cells = []
    worst_p1 = worst_seq = 0.0
    for cell in published:
        rc, qc = cell["ref_comment"], cell["query_comment"]
        if isinstance(rc, int):
            rmask = comments == rc
        else:
            rmask = np.isin(comments, rc) if len(rc) < 5 else np.ones_like(comments, dtype=bool)
        qmask = comments == qc
        # their subsampling is [::150] over the concatenated array in file order - reproduced exactly
        r_emb, r_lab = emb[rmask][::REF_EVERY], labels[rmask][::REF_EVERY]
        t1 = time.time()
        acc = calc.get_accuracy(emb[qmask], labels[qmask], r_emb, r_lab)
        p1 = np.array(acc["precision_at_1"])
        d1 = float(np.max(np.abs(p1 - np.array(cell["accuracies"]["precision_at_1"]))))
        rec = {"ref": rc, "query": qc, "n_query": int(qmask.sum()), "n_ref": int(len(r_lab)),
               "precision_at_1_max_abs_diff": d1, "precision_at_1_mean": float(p1.mean())}
        if seq:
            s10 = np.array(acc["sequence_top_1_accuracy_list_10_mins"])
            rec["seq10_max_abs_diff"] = float(np.max(np.abs(s10 - np.array(cell["accuracies"]["sequence_top_1_accuracy_list_10_mins"]))))
            worst_seq = max(worst_seq, rec["seq10_max_abs_diff"])
        worst_p1 = max(worst_p1, d1)
        cells.append(rec)
        print(f"  ref {rc} query {qc}: p@1 {p1.mean():.4f} maxdiff {d1:.2e}"
              + (f" seq10 maxdiff {rec['seq10_max_abs_diff']:.2e}" if seq else "") + f" ({time.time() - t1:.0f}s)", flush=True)
    single = [c for c in cells if isinstance(c["ref"], int)]
    out["calculator_gate"] = {
        "cells": cells, "worst_precision_at_1_abs_diff": worst_p1, "worst_seq10_abs_diff": worst_seq if seq else None,
        "within_mean": float(np.mean([c["precision_at_1_mean"] for c in single if c["ref"] == c["query"]])),
        "cross_mean": float(np.mean([c["precision_at_1_mean"] for c in single if c["ref"] != c["query"]])),
        "passed": bool(worst_p1 <= 1e-6 and (not seq or worst_seq <= 1e-6)),
        "pytorch_metric_learning": __import__("pytorch_metric_learning").__version__, "torch": torch.__version__,
    }
    print("calculator gate:", {k: v for k, v in out["calculator_gate"].items() if k != "cells"}, flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    # compact copy of their embeddings for the scoring stage (float32, derived from the release)
    np.savez_compressed(pathlib.Path(args.out).with_suffix(".npz"), embeddings=emb, label=labels, comments=comments)
    return 0 if out["calculator_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

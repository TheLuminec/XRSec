"""
Coordinator's acceptance criteria 1-3 for the amplitude / recorded-position baselines
(docs/COORDINATION.md at 3a388cf):

  1. On a `raw` checkpoint, `lookup_auc` and `lookup_auc_by_dataset` re-scored under the new
     code equal their recorded row to the digit, and `position_lookup_auc` equals them too.
  2. On a `dyn` checkpoint, `position_lookup_auc` per corpus equals the 9.10 per-axis
     harness's xyz lookup on the same seed-67 pairs, to the digit.
  3. `amplitude_auc` per corpus reproduces 9.14's amplitude table (its harness, seed 67) to
     the digit on the same checkpoint.

Pairs. A recorded row's evaluation manifest is drawn with `_seed_value(seed, 4)` (the test
set built beside training), and so were the 9.10 and 9.14 harnesses; `mode=test` draws with
`_seed_value(seed, 11)` and is therefore a different manifest by construction. To land on the
recorded pairs this script builds `SiameseDataset` directly with the training-time seed and
applies the checkpoint's normaliser exactly as the loader does, then runs `evaluate()`.

    CODE_ROOT=<worktree> python amplitude_baseline_criteria.py corpora <checkpoint> <out.json>
    CODE_ROOT=<worktree> python amplitude_baseline_criteria.py recorded <checkpoint> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\TheLu\Desktop\GIT\XRSec")
CODE_ROOT = Path(os.environ.get("CODE_ROOT", str(DATA_ROOT)))
sys.path.insert(0, str(CODE_ROOT / "model"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
torch.set_num_threads(4)

from dataset import SiameseDataset, _seed_value  # noqa: E402
from eval import evaluate  # noqa: E402
from metrics import roc_auc  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402

CORPORA = {"Head_and_Gaze": "Head_and_Gaze_Behavior_Dataset",
           "VR_User_Behavior": "VR_User_Behavior_Dataset_(Spherical_Video_Streaming)",
           "ViewGauss": "ViewGauss_Head-Movement_Dataset", "NJIT": "NJIT_6DOF_VR_Navigation_Dataset",
           "PanoSaliency": "360-degree_Saliency_Dataset_(PanoSaliency)", "Panonut360": "Panonut360_Dataset",
           "EyeNavGS": "EyeNavGS_6-DoF_Navigation_Dataset"}
SEED = 67
SHARD = DATA_ROOT / "results" / "runs" / "desktop-c.jsonl"


def _loader(ck, dirs, seed, exclude_users):
    es = ck["eval_split"]
    dataset = SiameseDataset(dirs, samples_per_user=512, sample_time=int(es["sample_time"]),
                             sample_rate=int(es["sample_rate"]), exclude_users=exclude_users, swap_data=False,
                             seed=_seed_value(seed, 4), within_dataset_negatives=True,
                             channels=ck.get("channels", "full"), window_stride=es.get("window_stride"),
                             cross_session_positives=True, encoding=es.get("encoding", "raw"))
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if normalizer.enabled:
        normalizer.transform(dataset.sample_index)
    return DataLoader(dataset, batch_size=256, shuffle=False)


def _evaluate(model, loader):
    _, _, metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"), return_metrics=True)
    return metrics


def corpora(checkpoint: str, out: str) -> None:
    """Criteria 2 and 3: one corpus at a time on the harnesses' seed-67 pairs."""
    model, ck = load_checkpoint(checkpoint, torch.device("cpu"), 100, return_checkpoint=True)
    es = ck["eval_split"]
    record = {"checkpoint": checkpoint, "encoding": es.get("encoding", "raw"), "sample_time": es["sample_time"],
              "window_stride": es.get("window_stride"), "manifest_seed": f"_seed_value({SEED}, 4)", "corpora": {}}
    failures = []
    for short, name in CORPORA.items():
        loader = _loader(ck, [str(DATA_ROOT / "processed_datasets" / name / "users")], SEED, [])
        metrics = _evaluate(model, loader)
        index, manifest = loader.dataset.sample_index, loader.dataset.manifest
        x1, x2, y = manifest["x1_indices"].view(-1), manifest["x2_indices"].view(-1), manifest["labels"].view(-1)
        pos = index.window_mean_positions                       # 9.10's harness: standardised means, xyz
        harness_xyz = float(roc_auc(-(pos[x1] - pos[x2]).norm(dim=1), y))
        amp = index.window_amplitudes                            # 9.14's harness: float64 sd-norm, pre-standardisation
        harness_amp = float(roc_auc(-(amp[x1] - amp[x2]).abs(), y))
        row = {"pairs": int(y.numel()), "users": int(index.num_users), "model_auc": metrics["auc"],
               "lookup_auc_encoded": metrics.get("lookup_auc"), "position_lookup_auc": metrics.get("position_lookup_auc"),
               "harness_xyz_lookup": harness_xyz, "amplitude_auc": metrics.get("amplitude_auc"), "harness_amplitude": harness_amp}
        record["corpora"][short] = row
        ok2 = repr(row["position_lookup_auc"]) == repr(harness_xyz)
        ok3 = repr(row["amplitude_auc"]) == repr(harness_amp)
        if not ok2:
            failures.append(f"{short}: position_lookup_auc {row['position_lookup_auc']!r} != harness xyz {harness_xyz!r}")
        if not ok3:
            failures.append(f"{short}: amplitude_auc {row['amplitude_auc']!r} != harness amplitude {harness_amp!r}")
        print(f"{short:<17} users {row['users']:>4} pairs {row['pairs']:>6}  model {row['model_auc']:.4f}  "
              f"lookup(encoded) {row['lookup_auc_encoded']:.4f}  position lookup {row['position_lookup_auc']:.6f} "
              f"vs harness {harness_xyz:.6f} {'OK' if ok2 else 'DIFF'}  amplitude {row['amplitude_auc']:.6f} "
              f"vs harness {harness_amp:.6f} {'OK' if ok3 else 'DIFF'}", flush=True)
    record["failures"] = failures
    Path(out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("CRITERIA 2 AND 3 " + ("PASSED: every corpus digit-exact against both harness formulas on the seed-67 pairs"
                                if not failures else "FAILED:\n  " + "\n  ".join(failures)))
    sys.exit(1 if failures else 0)


def recorded(checkpoint: str, out: str) -> None:
    """Criterion 1: the checkpoint's own recorded row, re-scored on its own evaluation set."""
    model, ck = load_checkpoint(checkpoint, torch.device("cpu"), 100, return_checkpoint=True)
    es = ck["eval_split"]
    rel = str(Path(checkpoint).resolve().relative_to(DATA_ROOT)).replace("\\", "/")
    row = None
    for line in SHARD.read_text(encoding="utf-8").splitlines():
        if line.strip() and (json.loads(line).get("checkpoint") or "").replace("\\", "/") == rel:
            row = json.loads(line)
    assert row is not None, f"no shard row for {rel}"
    recorded_by = {p.split("=")[0]: p.split("=")[1] for p in (row.get("lookup_auc_by_dataset") or "").split(";") if "=" in p}
    loader = _loader(ck, list(es["test_dirs"]), int(ck.get("seed", row["seed"])), list(es.get("exclude_users") or []))
    metrics = _evaluate(model, loader)
    record = {"checkpoint": checkpoint, "encoding": es.get("encoding", "raw"), "run_id": row["run_id"],
              "recorded": {"lookup_auc": row["lookup_auc"], "lookup_auc_by_dataset": recorded_by,
                           "selected_test_auc": row["selected_test_auc"]},
              "rescored": {"lookup_auc": metrics.get("lookup_auc"), "position_lookup_auc": metrics.get("position_lookup_auc"),
                           "amplitude_auc": metrics.get("amplitude_auc"), "auc_cpu": metrics["auc"],
                           "by_dataset": {n: {k: e.get(k) for k in ("lookup_auc", "position_lookup_auc", "amplitude_auc", "auc")}
                                          for n, e in metrics["by_dataset"].items()}}}
    problems = []
    if repr(metrics.get("lookup_auc")) != repr(row["lookup_auc"]):
        problems.append(f"pooled lookup_auc rescored {metrics.get('lookup_auc')!r} vs recorded {row['lookup_auc']!r}")
    for name, value in recorded_by.items():
        got = metrics["by_dataset"].get(name, {}).get("lookup_auc")
        if got is None or f"{got:.4f}" != value:
            problems.append(f"{name}: lookup_auc rescored {got} vs recorded {value}")
    gap = abs(float(metrics["position_lookup_auc"]) - float(metrics["lookup_auc"]))
    print(f"recorded row {row['run_id']} ({record['encoding']}): lookup_auc recorded {row['lookup_auc']!r} rescored "
          f"{metrics.get('lookup_auc')!r}; position_lookup_auc {metrics.get('position_lookup_auc')!r} (gap to lookup {gap:.1e}); "
          f"amplitude_auc {metrics.get('amplitude_auc')!r}; model AUC recorded (GPU) {row['selected_test_auc']:.6f} vs CPU {metrics['auc']:.6f}")
    for name in sorted(recorded_by):
        e = metrics["by_dataset"].get(name, {})
        print(f"  {name[:40]:<40} lookup recorded {recorded_by[name]} rescored {e.get('lookup_auc', float('nan')):.4f}  "
              f"position lookup {e.get('position_lookup_auc', float('nan')):.4f}  amplitude {e.get('amplitude_auc', float('nan')):.4f}")
    if record["encoding"] == "raw" and gap > 1e-6:
        problems.append(f"raw: position_lookup_auc differs from lookup_auc by {gap:.1e}")
    record["problems"] = problems
    Path(out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("CRITERION 1 " + ("PASSED: pooled lookup digit-exact, per-dataset lookups equal at the recorded precision, "
                            "recorded-position lookup equals the old one" if not problems else "FAILED:\n  " + "\n  ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    {"corpora": corpora, "recorded": recorded}[sys.argv[1]](sys.argv[2], sys.argv[3])

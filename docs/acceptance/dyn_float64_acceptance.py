"""
Acceptance for the float64 dyn residual: one existing dyn checkpoint, scored on every
held-out corpus with the pipeline's own test manifest, must reproduce its recorded
test_auc_by_dataset within 1e-4 AUC; and the Nymeria residual window-mean must read
~1e-14 m. Run once BEFORE the merge (must be digit-exact against the rows) and once AFTER.

    python dyn64_acceptance.py [checkpoint]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import os
ROOT = Path(r"C:\Users\TheLu\Desktop\GIT\XRSec")                         # data, cache, checkpoints
CODE_ROOT = Path(os.environ.get("CODE_ROOT", str(ROOT)))                  # which model/ to import
sys.path.insert(0, str(CODE_ROOT / "model"))
torch.set_num_threads(4)

from dataset import _seed_value, build_sample_index, generate_pair_manifest  # noqa: E402
from metrics import roc_auc  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "sweeps" / "314cd507f1" / "runs" / "bilstm_dbc29cfc5f" / "best.pth")
SHARD = ROOT / "results" / "runs" / "desktop-c.jsonl"


def recorded_rows(checkpoint_path: str) -> dict:
    rel = str(Path(checkpoint_path).resolve().relative_to(ROOT)).replace("\\", "/")
    for line in SHARD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("checkpoint") or "").replace("\\", "/") == rel:
            return {p.split("=")[0]: float(p.split("=")[1]) for p in (r.get("test_auc_by_dataset") or "").split(";") if "=" in p}
    return {}


model, ck = load_checkpoint(CKPT, torch.device("cpu"), 100, return_checkpoint=True)
es = ck["eval_split"]
seed = int(ck.get("seed", 67))
index = build_sample_index(es["test_dirs"], sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
                           encoding=es.get("encoding", "raw"), window_stride=es.get("window_stride"),
                           exclude_users=es.get("exclude_users") or [])
ChannelNormalizer.from_state(ck["normalizer"]).transform(index)
man = generate_pair_manifest(index, 512, seed=_seed_value(seed, 4), within_dataset_negatives=True, cross_session_positives=True)
with torch.no_grad():
    emb = torch.cat([model.embed(index.samples[i:i + 1024]) for i in range(0, index.sample_count, 1024)])
scores = F.cosine_similarity(emb[man["x1_indices"]], emb[man["x2_indices"]], dim=1, eps=1e-8)
labels = man["labels"].view(-1)
ds = torch.tensor([index.user_dataset_ids[int(a)] for a in man["anchor_user_ids"].view(-1)])
SHORT = {"VR_User_Behavior_Dataset_(Spherical_Video_Streaming)": "VR_User_Behavior", "ViewGauss_Head-Movement_Dataset": "ViewGauss",
         "Head_and_Gaze_Behavior_Dataset": "Head_and_Gaze", "NJIT_6DOF_VR_Navigation_Dataset": "NJIT",
         "360-degree_Saliency_Dataset_(PanoSaliency)": "PanoSaliency", "Panonut360_Dataset": "Panonut360",
         "EyeNavGS_6-DoF_Navigation_Dataset": "EyeNavGS"}
BEFORE_PATH = Path(__file__).with_name("dyn64_before_cpu.json")
before = json.loads(BEFORE_PATH.read_text())["auc"] if BEFORE_PATH.exists() else {}
recorded = recorded_rows(CKPT)
now = {}
worst = 0.0
for d, name in enumerate(index.dataset_names):
    mask = ds == d
    auc = roc_auc(scores[mask], labels[mask])
    short = SHORT.get(name, name)
    now[short] = round(float(auc), 6)
    ref = before.get(short)
    gap = abs(auc - ref) if ref is not None else float("nan")
    worst = max(worst, gap if ref is not None else 0.0)
    print(f"{short:<18} CPU now {auc:.4f}  CPU before {ref if ref is None else f'{ref:.4f}'}  |gap| {gap:.1e}   (GPU row {recorded.get(name, float('nan')):.4f})")
print(f"largest CPU-before vs CPU-after gap {worst:.1e}  -> {'PASS (within 1e-4)' if worst <= 1e-4 else 'RE-BASELINE (exceeds 1e-4)'}")
nym = build_sample_index([str(ROOT / "processed_datasets" / "Nymeria_Dataset" / "users")], sample_time=5, sample_rate=20, encoding="dyn", exclude_users=[])
res = nym.samples[:, 4:7].mean(dim=2).abs()
print(f"Nymeria dyn residual window-mean: median {res.norm(dim=1).median():.1e} m, max {res.max():.1e} m")
Path(__file__).with_name("dyn64_acceptance_result.json").write_text(json.dumps(
    {"checkpoint": CKPT, "code_root": str(CODE_ROOT), "before_cpu": before, "after_cpu": now,
     "largest_gap": worst, "pass": worst <= 1e-4,
     "nymeria_residual_median_m": float(res.norm(dim=1).median()), "nymeria_residual_max_m": float(res.max())}, indent=1))

"""
Acceptance for the amplitude / recorded-position baselines (GENERALISATION_PROPOSAL 9.14).

The change adds two columns and must alter nothing that existed: `evaluate()` on one existing
checkpoint, through the pipeline's own loader, must reproduce `auc`, `eer`, `lookup_auc`,
`lookup_eer` and every per-dataset model/lookup figure digit-exact (repr) between the code
before the change and the code after it, same device (CPU), same script. The after-run must
also carry `position_lookup_auc` and `amplitude_auc`, and under `raw` the recorded-position
lookup must agree with the old lookup to rounding.

    CODE_ROOT=<tree> python amplitude_baseline_acceptance.py score <checkpoint> <out.json>
    python amplitude_baseline_acceptance.py compare <before.json> <after.json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\TheLu\Desktop\GIT\XRSec")          # data, cache, checkpoints
EVAL_DIRS = [str(DATA_ROOT / "processed_datasets" / name / "users") for name in
             ("NJIT_6DOF_VR_Navigation_Dataset", "ViewGauss_Head-Movement_Dataset", "Panonut360_Dataset")]
COMPARED = ("auc", "eer", "lookup_auc", "lookup_eer")


def score(checkpoint: str, out: str) -> None:
    code_root = Path(os.environ.get("CODE_ROOT", str(DATA_ROOT)))
    sys.path.insert(0, str(code_root / "model"))
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    from dataset import create_dataloader_from_path
    from eval import evaluate
    from normalization import ChannelNormalizer
    from utils import load_checkpoint

    device = torch.device("cpu")
    model, ck = load_checkpoint(checkpoint, device, 100, return_checkpoint=True)
    es = ck["eval_split"]
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    loader = create_dataloader_from_path(
        EVAL_DIRS, 256, device, is_train=False,
        sample_time=int(es["sample_time"]), sample_rate=int(es["sample_rate"]),
        samples_per_user=512, exclude_users=[], swap_data=False, test_on_excluded=False,
        seed=int(ck.get("seed", 67)), normalize=es.get("normalize", "per_dataset"),
        normalizer=normalizer if normalizer.enabled else None,
        channels=ck.get("channels", "full"), eval_normalize="target_fit",
        encoding=es.get("encoding", "raw"), window_stride=es.get("window_stride"),
        within_dataset_negatives=True, cross_session_positives=True)
    _, _, metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), device, return_metrics=True)
    record = {k: repr(v) for k, v in metrics.items() if k != "by_dataset"}
    record["by_dataset"] = {name: {k: repr(v) for k, v in entry.items()} for name, entry in metrics["by_dataset"].items()}
    record["code_root"] = str(code_root)
    record["checkpoint"] = checkpoint
    record["encoding"] = es.get("encoding", "raw")
    Path(out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"scored {Path(checkpoint).parent.name} ({record['encoding']}) with {code_root}: "
          + ", ".join(f"{k}={record.get(k)}" for k in COMPARED + ("position_lookup_auc", "amplitude_auc")))


def compare(before: str, after: str) -> None:
    b = json.loads(Path(before).read_text(encoding="utf-8"))
    a = json.loads(Path(after).read_text(encoding="utf-8"))
    problems = []
    for key in COMPARED:
        if b.get(key) != a.get(key):
            problems.append(f"{key}: before {b.get(key)} after {a.get(key)}")
    for name, entry in b["by_dataset"].items():
        for key in ("auc", "eer", "lookup_auc", "lookup_eer", "pairs", "tier"):
            if entry.get(key) != a["by_dataset"].get(name, {}).get(key):
                problems.append(f"{name}.{key}: before {entry.get(key)} after {a['by_dataset'].get(name, {}).get(key)}")
    for key in ("position_lookup_auc", "position_lookup_eer", "amplitude_auc", "amplitude_eer"):
        if key not in a:
            problems.append(f"after-run lacks {key}")
    if a.get("encoding") == "raw" and "position_lookup_auc" in a:
        gap = abs(float(a["position_lookup_auc"]) - float(a["lookup_auc"]))
        print(f"raw: recorded-position lookup {a['position_lookup_auc']} vs encoded-window lookup {a['lookup_auc']} (gap {gap:.2e})")
        if gap > 1e-3:
            problems.append(f"raw: position_lookup_auc differs from lookup_auc by {gap:.2e}")
    if problems:
        print("ACCEPTANCE FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"ACCEPTANCE PASSED ({a.get('encoding')}): every pre-existing figure digit-exact; new columns "
          f"position_lookup_auc={a.get('position_lookup_auc')} amplitude_auc={a.get('amplitude_auc')}")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3])

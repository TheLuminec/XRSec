"""
Score checkpoints on Nymeria (never in any training set) and log a shard row per checkpoint.

Every other figure in this repository has a row in results/runs/<machine>.jsonl that anyone
can re-derive in one command; until this script, Nymeria figures came from scratchpad scoring
and had none. Scoring goes through the pipeline's own SiameseDataset + evaluate(), so the row
carries the same columns as a training run - model AUC/EER, the encoded-window lookup, the
recorded-position lookup, movement amplitude, per-dataset split - on the manifest the pipeline
would draw for that checkpoint (`_seed_value(seed, 4)` from the checkpoint's own seed).

    python score_nymeria.py <checkpoint.pth> [...]         one row per checkpoint
    python score_nymeria.py --sweep <sweep_id> [...]       every best.pth under sweeps/<id>/runs
    python score_nymeria.py --all-dyn                      every dyn transfer row's checkpoint in this machine's shard

CPU by default (the recorded CPU figures in the proposal were CPU); DEVICE=cuda to override.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

import results_log  # noqa: E402
from dataset import SiameseDataset, _seed_value  # noqa: E402
from eval import evaluate, format_by_dataset  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402

NYMERIA = str(ROOT / "processed_datasets" / "Nymeria_Dataset" / "users")
DEVICE = torch.device(os.environ.get("DEVICE", "cpu"))


def score(ckpt_path: str) -> dict:
    model, ck = load_checkpoint(ckpt_path, DEVICE, 100, return_checkpoint=True)
    es = ck["eval_split"]
    seed = int(ck.get("seed", 67))
    st, sr = int(es["sample_time"]), int(es["sample_rate"])
    dataset = SiameseDataset(NYMERIA, samples_per_user=512, sample_time=st, sample_rate=sr, exclude_users=[],
                             swap_data=False, seed=_seed_value(seed, 4), within_dataset_negatives=True,
                             channels=ck.get("channels", "full"), window_stride=es.get("window_stride"),
                             cross_session_positives=True, encoding=es.get("encoding", "raw"))
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if normalizer.enabled:
        normalizer.transform(dataset.sample_index)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    _, accuracy, metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
    labels = dataset.manifest["labels"].view(-1)
    history = {
        "selected_test_auc": metrics["auc"], "selected_test_eer": metrics["eer"],
        "best_test_auc": metrics["auc"], "best_test_eer": metrics["eer"], "selected_test_acc": accuracy,
        "lookup_auc": metrics.get("lookup_auc"), "lookup_eer": metrics.get("lookup_eer"),
        "position_lookup_auc": metrics.get("position_lookup_auc"), "position_lookup_eer": metrics.get("position_lookup_eer"),
        "amplitude_auc": metrics.get("amplitude_auc"), "amplitude_eer": metrics.get("amplitude_eer"),
        "selected_test_by_dataset": metrics.get("by_dataset") or {},
        "unseen_datasets": dict(normalizer.unseen_datasets),
        "eval_positive_fraction": float(labels.float().mean()), "best_epoch": int(ck.get("epoch", 0)),
    }
    cfg = SimpleNamespace(
        mode="rescore", experiment_name="nymeria_rescored", extractor=ck.get("extractor"),
        extractor_params=ck.get("extractor_params") or None, objective=ck.get("objective"),
        identity_margin=0.35, identity_scale=30.0, balance_identities=False, balance_cap=None,
        head=ck.get("head"), channels=ck.get("channels", "full"), encoding=es.get("encoding", "raw"),
        resample=es.get("resample", "nearest"), window_stride=es.get("window_stride"),
        sweep_id=Path(ckpt_path).resolve().parents[2].name if "sweeps" in Path(ckpt_path).resolve().parts else "",
        fold="", normalize=es.get("normalize", "per_dataset"), within_dataset_negatives=True,
        cross_session_positives=True, center_position=False, max_users=es.get("max_users"),
        eval_normalize="target_fit", seed=seed, sample_time=st, sample_rate=sr,
        embedding_dim=int(ck.get("embedding_dim", 128)), samples_per_user=512, batch_size=256, lr=0.001,
        weight_decay=0.0, val_user_fraction=0.25, data_dirs=list(es.get("data_dirs") or []), test_dirs=[NYMERIA],
        exclude_users=[], swap_data=False, test_on_excluded=False, model_path=ckpt_path, save_path=ckpt_path,
        boosting=None,
    )
    path = results_log.append_run(cfg, history, dataset_tag="nymeria")
    print(f"{Path(ckpt_path).parent.name}: {es.get('encoding', 'raw')} {st}s stride {es.get('window_stride')} "
          f"max_users={es.get('max_users')} seed {seed}  Nymeria AUC {metrics['auc']:.4f}  "
          f"lookup(encoded) {metrics.get('lookup_auc', float('nan')):.4f}  position lookup "
          f"{metrics.get('position_lookup_auc', float('nan')):.4f}  amplitude {metrics.get('amplitude_auc', float('nan')):.4f}  "
          f"users {dataset.sample_index.num_users} pairs {int(labels.numel())}  -> {path}", flush=True)
    return metrics


def checkpoints_from_args(args: list[str]) -> list[str]:
    if args and args[0] == "--all-dyn":
        rows = [json.loads(l) for l in (ROOT / "results" / "runs" / f"{results_log.machine_name()}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        seen, out = set(), []
        for r in rows:
            if r.get("experiment") == "transfer" and r.get("encoding") == "dyn" and r.get("checkpoint"):
                p = str(ROOT / r["checkpoint"])
                if p not in seen and os.path.exists(p):
                    seen.add(p)
                    out.append(p)
        return out
    if args and args[0] == "--sweep":
        return sorted(p for s in args[1:] for p in glob.glob(str(ROOT / "sweeps" / s / "runs" / "*" / "best.pth")))
    return args


if __name__ == "__main__":
    targets = checkpoints_from_args(sys.argv[1:])
    print(f"{len(targets)} checkpoint(s) on {DEVICE}")
    for ckpt in targets:
        score(ckpt)

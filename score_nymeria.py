"""
Score checkpoints on Nymeria (never in any training set) and log a shard row per checkpoint.

Every other figure in this repository has a row in results/runs/<machine>.jsonl that anyone
can re-derive in one command; until this script, Nymeria figures came from scratchpad scoring
and had none. Scoring goes through the pipeline's own SiameseDataset + evaluate(), so the row
carries the same columns as a training run - model AUC/EER, the encoded-window lookup, the
recorded-position lookup, movement amplitude, per-dataset split - on the manifest the pipeline
would draw for that checkpoint (`_seed_value(seed, 4)` from the checkpoint's own seed).

The gate (Coordinator, 2026-09-05, after the void step 6 columns): before a checkpoint is
scored outside the training path, it must reproduce its own recorded `selected_test_auc` on
its own recorded evaluation users. `--gate` runs that first and refuses to log a Nymeria row
for a checkpoint that fails it; `--gate-only` runs only the gates and writes
docs/acceptance/nymeria_gate.json. The gate runs on GATE_DEVICE (cuda when available: the rows
were written on the GPU and cuDNN reproduces them to ~1e-6 across runs, CPU only to ~7e-4).

    python score_nymeria.py <checkpoint.pth> [...]         one row per checkpoint
    python score_nymeria.py --sweep <sweep_id> [...]       every best.pth under sweeps/<id>/runs
    python score_nymeria.py --all-dyn                      every dyn transfer row's checkpoint in this machine's shard
    python score_nymeria.py --gate --all-dyn               gate first, score only what passes
    python score_nymeria.py --gate-only --all-dyn          gates only, JSON record

Nymeria scoring is CPU by default (the recorded CPU figures in the proposal were CPU);
DEVICE=cuda overrides it.
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
from eval import evaluate  # noqa: E402
from normalization import ChannelNormalizer  # noqa: E402
from utils import load_checkpoint  # noqa: E402

NYMERIA = str(ROOT / "processed_datasets" / "Nymeria_Dataset" / "users")
DEVICE = torch.device(os.environ.get("DEVICE", "cpu"))
GATE_DEVICE = torch.device(os.environ.get("GATE_DEVICE", "cuda" if torch.cuda.device_count() else "cpu"))
GATE_TOLERANCE = {"cuda": 1e-4, "cpu": 2e-3}          # cuDNN across runs ~1e-6; CPU vs GPU up to 7e-4
GATE_RECORD = ROOT / "docs" / "acceptance" / "nymeria_gate.json"


def _shard_rows() -> list[dict]:
    path = ROOT / "results" / "runs" / f"{results_log.machine_name()}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _recorded_row(ckpt_path: str) -> dict | None:
    rel = str(Path(ckpt_path).resolve().relative_to(ROOT)).replace("\\", "/")
    rows = [r for r in _shard_rows() if (r.get("checkpoint") or "").replace("\\", "/") == rel
            and r.get("mode") in (None, "train", "sweep") and r.get("experiment") != "nymeria_rescored"]
    return rows[-1] if rows else None


def _loader(ck: dict, dirs, seed: int, exclude_users) -> DataLoader:
    es = ck["eval_split"]
    dataset = SiameseDataset(dirs, samples_per_user=512, sample_time=int(es["sample_time"]),
                             sample_rate=int(es["sample_rate"]), exclude_users=exclude_users, swap_data=False,
                             seed=_seed_value(seed, 4), within_dataset_negatives=True,
                             channels=ck.get("channels", "full"), window_stride=es.get("window_stride"),
                             cross_session_positives=True, encoding=es.get("encoding", "raw"))
    normalizer = ChannelNormalizer.from_state(ck.get("normalizer"), unseen="target_fit")
    if normalizer.enabled:
        normalizer.transform(dataset.sample_index)
    dataset.unseen_datasets = dict(normalizer.unseen_datasets)
    return DataLoader(dataset, batch_size=256, shuffle=False)


def gate(ckpt_path: str) -> dict:
    """Reproduce the checkpoint's recorded selected_test_auc on its recorded evaluation users."""
    row = _recorded_row(ckpt_path)
    if row is None:
        return {"checkpoint": ckpt_path, "passed": False, "reason": "no training row in the shard for this checkpoint"}
    model, ck = load_checkpoint(ckpt_path, GATE_DEVICE, 100, return_checkpoint=True)
    es = ck["eval_split"]
    loader = _loader(ck, list(es["test_dirs"]), int(ck.get("seed", row["seed"])), list(es.get("exclude_users") or []))
    _, _, metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), GATE_DEVICE, return_metrics=True)
    gap = abs(float(metrics["auc"]) - float(row["selected_test_auc"]))
    tolerance = GATE_TOLERANCE[GATE_DEVICE.type]
    result = {"checkpoint": ckpt_path, "run_id": row["run_id"], "recorded": row["selected_test_auc"],
              "rescored": metrics["auc"], "gap": gap, "device": str(GATE_DEVICE), "tolerance": tolerance,
              "lookup_recorded": row.get("lookup_auc"), "lookup_rescored": metrics.get("lookup_auc"),
              "passed": gap <= tolerance}
    print(f"gate {Path(ckpt_path).parent.name} ({es.get('encoding', 'raw')} {es['sample_time']}s seed {row['seed']}): "
          f"recorded {row['selected_test_auc']:.6f} rescored {metrics['auc']:.6f} on {GATE_DEVICE} gap {gap:.1e} "
          f"{'PASS' if result['passed'] else 'FAIL'}", flush=True)
    return result


def score(ckpt_path: str) -> dict:
    model, ck = load_checkpoint(ckpt_path, DEVICE, 100, return_checkpoint=True)
    es = ck["eval_split"]
    seed = int(ck.get("seed", 67))
    st, sr = int(es["sample_time"]), int(es["sample_rate"])
    loader = _loader(ck, NYMERIA, seed, [])
    dataset = loader.dataset
    _, accuracy, metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), DEVICE, return_metrics=True)
    labels = dataset.manifest["labels"].view(-1)
    history = {
        "selected_test_auc": metrics["auc"], "selected_test_eer": metrics["eer"],
        "best_test_auc": metrics["auc"], "best_test_eer": metrics["eer"], "selected_test_acc": accuracy,
        "lookup_auc": metrics.get("lookup_auc"), "lookup_eer": metrics.get("lookup_eer"),
        "position_lookup_auc": metrics.get("position_lookup_auc"), "position_lookup_eer": metrics.get("position_lookup_eer"),
        "amplitude_auc": metrics.get("amplitude_auc"), "amplitude_eer": metrics.get("amplitude_eer"),
        "selected_test_by_dataset": metrics.get("by_dataset") or {},
        "unseen_datasets": getattr(dataset, "unseen_datasets", {}),
        "eval_positive_fraction": float(labels.float().mean()), "best_epoch": int(ck.get("epoch", 0)),
    }
    resolved = Path(ckpt_path).resolve()
    sweep_id = resolved.parents[2].name if "sweeps" in resolved.parts else ""
    cfg = SimpleNamespace(
        mode="rescore", experiment_name="nymeria_rescored", extractor=ck.get("extractor"),
        extractor_params=ck.get("extractor_params") or None, objective=ck.get("objective"),
        identity_margin=0.35, identity_scale=30.0, balance_identities=False, balance_cap=None,
        head=ck.get("head"), channels=ck.get("channels", "full"), encoding=es.get("encoding", "raw"),
        resample=es.get("resample", "nearest"), window_stride=es.get("window_stride"),
        sweep_id=sweep_id, fold="", normalize=es.get("normalize", "per_dataset"), within_dataset_negatives=True,
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
        seen, out = set(), []
        for r in _shard_rows():
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
    args = sys.argv[1:]
    do_gate = "--gate" in args or "--gate-only" in args
    gate_only = "--gate-only" in args
    args = [a for a in args if a not in ("--gate", "--gate-only")]
    targets = checkpoints_from_args(args)
    print(f"{len(targets)} checkpoint(s); scoring on {DEVICE}; gate {'on ' + str(GATE_DEVICE) if do_gate else 'off'}")
    records = []
    for ckpt in targets:
        if do_gate:
            result = gate(ckpt)
            records.append(result)
            if not result["passed"]:
                print(f"  refusing to score {ckpt}: {result.get('reason', 'gate failed')}", flush=True)
                continue
        if not gate_only:
            score(ckpt)
    if do_gate:
        GATE_RECORD.parent.mkdir(parents=True, exist_ok=True)
        GATE_RECORD.write_text(json.dumps(records, indent=1), encoding="utf-8")
        passed = sum(r["passed"] for r in records)
        print(f"gate: {passed}/{len(records)} passed -> {GATE_RECORD}")

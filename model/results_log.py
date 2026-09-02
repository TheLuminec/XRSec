"""
Append-only results log.

Every run prints its metrics to stdout and nothing else, so Hydra's per-run
``main.log`` files are empty and past results survive only inside checkpoint
``history`` dicts and PNG plots. This module writes one row per run to a single
CSV at the repo root so experiments can be compared by sorting a table instead of
reopening checkpoints.

Logging must never take down a run that has already done the expensive work, so
every failure here is caught and downgraded to a warning.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "runs.csv"

# Fixed column order. Append new columns at the end so existing files stay readable.
FIELDS = [
    "timestamp",
    "mode",
    "boosting",
    "experiment",
    "dataset_tag",
    "extractor",
    "extractor_params",
    "objective",
    "head",
    "sweep_id",
    "normalize",
    "within_dataset_negatives",
    "best_test_acc",
    "best_test_auc",
    "best_test_eer",
    "best_epoch",
    "final_train_acc",
    "final_test_acc",
    "final_train_loss",
    "final_test_loss",
    "epochs_run",
    "rounds_run",
    "seed",
    "sample_time",
    "sample_rate",
    "seq_len",
    "embedding_dim",
    "samples_per_user",
    "batch_size",
    "lr",
    "weight_decay",
    "hard_fraction",
    "candidate_pairs_per_user",
    "match_ratio",
    "num_data_dirs",
    "num_test_dirs",
    "num_excluded_users",
    "swap_data",
    "test_on_excluded",
    "data_dirs",
    "test_dirs",
    "checkpoint",
    "run_dir",
    "git_sha",
]


def _git_sha() -> str:
    """Short SHA plus a dirty marker, or empty string outside a usable git tree."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode != 0:
            return ""
        revision = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            revision += "-dirty"
        return revision
    except Exception:
        return ""


def _relative_to_repo(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def _names(paths) -> str:
    """Dataset roots as a compact `a|b` string of dataset names, not full paths."""
    out = []
    for path in paths or []:
        candidate = Path(path)
        # Roots are .../<Dataset_Name>/users, so the dataset name is one level up.
        name = candidate.parent.name if candidate.name == "users" else candidate.name
        out.append(name)
    return "|".join(out)


def _params(params) -> str:
    """Extractor hyperparameters as a stable `k=v;k=v` string, sortable in a spreadsheet."""
    try:
        if params is None:
            return ""
        items = params.items() if hasattr(params, "items") else dict(params).items()
        return ";".join(f"{key}={value}" for key, value in sorted(items))
    except Exception:
        return str(params)


def _last(values):
    return values[-1] if values else None


def summarize(mode: str, result) -> dict:
    """Flatten a train/boosted/test result into the metric columns."""
    if mode == "test":
        loss, accuracy, *rest = result
        metrics = rest[0] if rest else {}
        return {"final_test_loss": loss, "final_test_acc": accuracy, "best_test_acc": accuracy,
                "best_test_auc": metrics.get("auc"), "best_test_eer": metrics.get("eer")}

    if isinstance(result, dict) and result.get("mode") == "boosted":
        summaries = result.get("round_summaries") or []
        histories = result.get("round_histories") or []
        best = max(summaries, key=lambda s: s["best_test_acc"], default=None)
        final = histories[-1] if histories else {}
        return {
            "best_test_acc": best["best_test_acc"] if best else None,
            "best_test_auc": (histories[best["round_idx"]].get("best_test_auc")
                              if best and "round_idx" in best and best["round_idx"] < len(histories) else None),
            "best_epoch": best["best_epoch"] if best else None,
            "rounds_run": len(summaries),
            "epochs_run": len(final.get("train_loss") or []),
            "final_train_acc": _last(final.get("train_acc")),
            "final_test_acc": _last(final.get("test_acc")),
            "final_train_loss": _last(final.get("train_loss")),
            "final_test_loss": _last(final.get("test_loss")),
            "checkpoint": _relative_to_repo(result.get("best_checkpoint")),
        }

    history = result if isinstance(result, dict) else {}
    return {
        "best_test_acc": history.get("best_test_acc"),
        "best_test_auc": history.get("best_test_auc"),
        "best_test_eer": history.get("best_test_eer"),
        "best_epoch": history.get("best_epoch"),
        "epochs_run": len(history.get("train_loss") or []),
        "final_train_acc": _last(history.get("train_acc")),
        "final_test_acc": _last(history.get("test_acc")),
        "final_train_loss": _last(history.get("train_loss")),
        "final_test_loss": _last(history.get("test_loss")),
    }


def _existing_header(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), None)


def _append_row(path: Path, row: dict) -> None:
    """
    Append one row, migrating the file in place if the schema has changed.

    Columns get added over time (new metrics, new sweep axes). Appending wider rows
    under a stale header silently corrupts the file, so when the header no longer
    matches, the whole file is rewritten with the new columns and existing rows are
    preserved with blanks. Columns that exist only in the old file are kept at the
    end rather than dropped.
    """
    header = _existing_header(path)

    if header is None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    if header == FIELDS:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore").writerow(row)
        return

    fieldnames = FIELDS + [column for column in header if column not in FIELDS]
    with path.open("r", newline="", encoding="utf-8") as handle:
        existing = [
            {key: value for key, value in record.items() if key is not None}
            for record in csv.DictReader(handle)
            # `results/runs.csv` is union-merged across machines (.gitattributes), so a
            # merge of two differing schemas can leave a second header line mid-file.
            # Treat it as the artefact it is rather than a row of data.
            if record.get(FIELDS[0]) != FIELDS[0]
        ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    print(f"Results log schema updated ({len(header)} -> {len(fieldnames)} columns)")


def append_run(cfg, result, dataset_tag: str, results_path: Path | None = None) -> Path | None:
    """Append one row describing this run. Returns the path written, or None on failure."""
    try:
        path = Path(results_path) if results_path else DEFAULT_RESULTS_PATH
        boosting = getattr(cfg, "boosting", None)
        boosting_enabled = bool(getattr(boosting, "enabled", False)) if boosting else False
        mode = str(cfg.mode)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "boosting": boosting_enabled,
            "experiment": cfg.experiment_name,
            "dataset_tag": dataset_tag,
            "extractor": getattr(cfg, "extractor", ""),
            "objective": getattr(cfg, "objective", ""),
            "head": getattr(cfg, "head", ""),
            "extractor_params": _params(getattr(cfg, "extractor_params", None)),
            "sweep_id": getattr(cfg, "sweep_id", ""),
            "normalize": getattr(cfg, "normalize", ""),
            "within_dataset_negatives": getattr(cfg, "within_dataset_negatives", ""),
            "seed": cfg.seed,
            "sample_time": cfg.sample_time,
            "sample_rate": cfg.sample_rate,
            "seq_len": int(cfg.sample_time) * int(cfg.sample_rate),
            "embedding_dim": cfg.embedding_dim,
            "samples_per_user": cfg.samples_per_user,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": getattr(cfg, "weight_decay", ""),
            "epochs_run": None,
            "rounds_run": None,
            "num_data_dirs": len(cfg.data_dirs or []),
            "num_test_dirs": len(cfg.test_dirs or []),
            "num_excluded_users": len(getattr(cfg, "exclude_users", None) or []),
            "swap_data": getattr(cfg, "swap_data", None),
            "test_on_excluded": getattr(cfg, "test_on_excluded", None),
            "data_dirs": _names(cfg.data_dirs),
            "test_dirs": _names(cfg.test_dirs),
            "checkpoint": _relative_to_repo(cfg.model_path if mode == "test" else cfg.save_path),
            "run_dir": _relative_to_repo(Path.cwd()),
            "git_sha": _git_sha(),
        }
        if boosting_enabled:
            row.update({
                "hard_fraction": boosting.hard_fraction,
                "candidate_pairs_per_user": boosting.candidate_pairs_per_user,
                "match_ratio": boosting.match_ratio,
            })
        row.update(summarize(mode, result))

        path.parent.mkdir(parents=True, exist_ok=True)
        _append_row(path, row)

        print(f"Run recorded in {path}")
        return path
    except Exception as exc:
        print(f"WARNING: could not append to results log: {exc}")
        return None

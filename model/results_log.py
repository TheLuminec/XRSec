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
import hashlib
import json
import platform
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

#: Frozen history. Every run up to the switch to per-machine shards lives here and
#: nothing appends to it any more.
LEGACY_RESULTS_PATH = REPO_ROOT / "results" / "runs.csv"

#: What runs are actually written to: one append-only JSONL per machine.
#:
#: The log is committed from three machines and merged with `merge=union`
#: (.gitattributes), which unions *lines*. A CSV's meaning lives in a header those
#: lines share, and this schema migrates by design - so the moment two machines hold
#: different column counts, union files every row from one side under the other's
#: header. That happened: 537 rows, 237 duplicated, seed 67 reading as 2, run_dir
#: holding a git SHA. Both inputs were individually clean.
#:
#: JSONL removes the class rather than patching it. Each line is a self-describing
#: record, so a union of any two versions is correct by construction, adding a field
#: is a non-event, and appending never rewrites an existing line. Sharding per machine
#: means the common case does not merge at all.
RUNS_DIR = REPO_ROOT / "results" / "runs"

#: Derived, gitignored, rebuilt after every append: the whole log as one table for
#: analysis. Read this, never write it.
COMBINED_RESULTS_PATH = REPO_ROOT / "results" / "runs_all.csv"

DEFAULT_RESULTS_PATH = LEGACY_RESULTS_PATH  # retained: older callers import this name

# Fixed column order. Append new columns at the end so existing files stay readable.
FIELDS = [
    "run_id",
    "timestamp",
    "mode",
    "boosting",
    "experiment",
    "dataset_tag",
    "extractor",
    "extractor_params",
    "objective",
    "identity_margin",
    "identity_scale",
    "balance_identities",
    "balance_cap",
    "head",
    "channels",
    "encoding",
    "resample",
    "window_stride",
    "sweep_id",
    "normalize",
    "within_dataset_negatives",
    "cross_session_positives",
    "same_session_fallback_users",
    "eval_positive_fraction",
    "template_k",
    "rank1",
    "gallery_users",
    "center_position",
    "max_users",
    "selected_test_acc",
    "best_test_acc",
    "best_val_acc",
    "selected_test_auc",
    "selected_test_eer",
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
    "val_user_fraction",
    "hard_fraction",
    "candidate_pairs_per_user",
    "match_ratio",
    "num_data_dirs",
    "num_train_identities",
    "num_test_dirs",
    "num_excluded_users",
    "swap_data",
    "test_on_excluded",
    "data_dirs",
    "test_dirs",
    "checkpoint",
    "run_dir",
    "git_sha",
    "code_identity",
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


_CODE_ROOT = REPO_ROOT / "model"
_code_identity_cache: str | None = None


def code_identity() -> str:
    """
    Content hash of every .py under model/, as a short digest.

    A config digest is not enough to identify an experiment: the code that turns
    config into numbers is half of it. A bugfix changes no config key, so a sweep
    resumed after one matched its old state file, skipped every run, and re-printed
    the pre-fix numbers under the new banner. That cost three launches in one day and
    it fails in the direction of confirming whatever was measured before.

    Content rather than the git SHA, because the tree is routinely dirty and every
    dirty state would otherwise share one identity - which is exactly the collision
    being prevented. A comment-only edit does invalidate resume; that trade is
    deliberate, since a false reuse costs a wrong answer and a false invalidation
    costs recompute.
    """
    global _code_identity_cache
    if _code_identity_cache is not None:
        return _code_identity_cache

    try:
        digest = hashlib.sha1()
        for path in sorted(_CODE_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(_CODE_ROOT)).replace("\\", "/").encode("utf-8"))
            digest.update(hashlib.sha1(path.read_bytes()).digest())
        _code_identity_cache = digest.hexdigest()[:10]
    except Exception:
        _code_identity_cache = ""
    return _code_identity_cache


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
    """Flatten a train/boosted/test/curve result into the metric columns."""
    if mode == "curve":
        # One row per k. A curve that only ever existed in stdout would repeat the
        # mistake this file was written to fix.
        return {
            "template_k": result.get("k"),
            "selected_test_auc": result.get("auc"),
            "best_test_auc": result.get("auc"),
            "selected_test_eer": result.get("eer"),
            "best_test_eer": result.get("eer"),
            "selected_test_acc": result.get("accuracy_at_eer"),
            "eval_positive_fraction": result.get("positive_fraction"),
        }

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
        "selected_test_acc": history.get("selected_test_acc"),
        "same_session_fallback_users": history.get("same_session_fallback_users"),
        "eval_positive_fraction": history.get("eval_positive_fraction"),
        "best_val_acc": history.get("best_val_acc"),
        "best_test_auc": history.get("best_test_auc"),
        "selected_test_auc": history.get("selected_test_auc"),
        "selected_test_eer": history.get("selected_test_eer"),
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


def machine_name() -> str:
    """Filesystem-safe host name, used to give each machine its own shard."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", platform.node() or "unknown").strip("-")
    return name.lower() or "unknown"


def shard_path() -> Path:
    return RUNS_DIR / f"{machine_name()}.jsonl"


def _append_jsonl(path: Path, row: dict) -> None:
    """
    Append one self-describing record. Never rewrites an existing line.

    That restriction is the whole point: it is what makes `merge=union` correct for
    this file instead of merely conflict-free.
    """
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # One damaged line must not hide every other run in the file.
            print(f"WARNING: skipping unreadable line {number} of {path.name}")
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def load_runs(include_legacy: bool = True) -> list[dict]:
    """Every recorded run, from the frozen CSV and every machine's shard."""
    rows: list[dict] = []
    if include_legacy and LEGACY_RESULTS_PATH.exists():
        with LEGACY_RESULTS_PATH.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(
                {key: value for key, value in record.items() if key is not None}
                for record in csv.DictReader(handle)
                if record.get("timestamp") != "timestamp"
            )
    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("*.jsonl")):
            rows.extend(_read_jsonl(path))
    rows.sort(key=lambda row: str(row.get("timestamp") or ""))
    return rows


def write_combined_csv(path: Path | None = None) -> Path:
    """
    Materialise the whole log as one table. Derived and gitignored - the shards are
    the record, this is the convenient view of it.
    """
    path = Path(path) if path else COMBINED_RESULTS_PATH
    rows = load_runs()
    extra = [key for row in rows for key in row if key not in FIELDS]
    fieldnames = FIELDS + sorted(dict.fromkeys(extra))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_run(cfg, result, dataset_tag: str, results_path: Path | None = None) -> Path | None:
    """Append one row describing this run. Returns the path written, or None on failure."""
    try:
        path = Path(results_path) if results_path else shard_path()
        boosting = getattr(cfg, "boosting", None)
        boosting_enabled = bool(getattr(boosting, "enabled", False)) if boosting else False
        mode = str(cfg.mode)

        row = {
            # Makes every line unique, so a union merge can never coalesce two
            # distinct runs that happen to agree on all their fields.
            "run_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "boosting": boosting_enabled,
            "experiment": cfg.experiment_name,
            "dataset_tag": dataset_tag,
            "extractor": getattr(cfg, "extractor", ""),
            "objective": getattr(cfg, "objective", ""),
            "identity_margin": getattr(cfg, "identity_margin", ""),
            "identity_scale": getattr(cfg, "identity_scale", ""),
            "balance_identities": getattr(cfg, "balance_identities", ""),
            "balance_cap": getattr(cfg, "balance_cap", ""),
            "head": getattr(cfg, "head", ""),
            "channels": getattr(cfg, "channels", ""),
            "encoding": getattr(cfg, "encoding", ""),
            "resample": getattr(cfg, "resample", ""),
            "window_stride": getattr(cfg, "window_stride", ""),
            "extractor_params": _params(getattr(cfg, "extractor_params", None)),
            "sweep_id": getattr(cfg, "sweep_id", ""),
            "normalize": getattr(cfg, "normalize", ""),
            "within_dataset_negatives": getattr(cfg, "within_dataset_negatives", ""),
            "cross_session_positives": getattr(cfg, "cross_session_positives", ""),
            "center_position": getattr(cfg, "center_position", ""),
            "max_users": getattr(cfg, "max_users", ""),
            "seed": cfg.seed,
            "sample_time": cfg.sample_time,
            "sample_rate": cfg.sample_rate,
            "seq_len": int(cfg.sample_time) * int(cfg.sample_rate),
            "embedding_dim": cfg.embedding_dim,
            "samples_per_user": cfg.samples_per_user,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": getattr(cfg, "weight_decay", ""),
            "val_user_fraction": getattr(cfg, "val_user_fraction", ""),
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
            "code_identity": code_identity(),
        }
        if boosting_enabled:
            row.update({
                "hard_fraction": boosting.hard_fraction,
                "candidate_pairs_per_user": boosting.candidate_pairs_per_user,
                "match_ratio": boosting.match_ratio,
            })
        row.update(summarize(mode, result))

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            _append_jsonl(path, row)
            # Keep the readable table current without ever committing it.
            try:
                write_combined_csv()
            except Exception as exc:
                print(f"WARNING: could not rebuild {COMBINED_RESULTS_PATH.name}: {exc}")
        else:
            _append_row(path, row)

        print(f"Run recorded in {path}")
        return path
    except Exception as exc:
        print(f"WARNING: could not append to results log: {exc}")
        return None

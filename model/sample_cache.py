"""
On-disk cache for sampled motion windows.

Reading a user's CSVs and running them through ``Sampler`` is the dominant cost of
starting a run: roughly 60% ``pd.read_csv`` and 40% the window loop, and both the
standard and boosted paths build a sample index twice (train + eval). None of that
work depends on which users are excluded, so it is cached once per user directory
and reused by every later run, split, and boosting round.

Cache entries are keyed by the content of the user directory (CSV names, sizes and
mtimes) plus the sampling parameters, so editing or regenerating a dataset
invalidates the affected entries automatically.

This is only safe because sampling is deterministic: ``UserProfile`` builds every
``Sampler`` with ``index_randomness=0``. If per-epoch index jitter is ever enabled,
the cache would freeze one fixed augmentation and must be bypassed.

Set ``XRSEC_SAMPLE_CACHE=0`` to disable.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch


CACHE_VERSION = 1

# Anchored to the repo root, never the cwd: Hydra runs with `job.chdir: true`, so a
# relative cache directory would land inside each run directory and never be reused.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "samples"


def cache_enabled() -> bool:
    return os.environ.get("XRSEC_SAMPLE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def cache_dir() -> Path:
    override = os.environ.get("XRSEC_SAMPLE_CACHE_DIR")
    return Path(override) if override else DEFAULT_CACHE_DIR


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:48] or "x"


def directory_signature(user_dir: Path, sample_time: int, sample_rate: int) -> str:
    """
    Content signature for one user directory at one sampling resolution.

    Uses CSV name/size/mtime rather than file contents so the check stays cheap
    (a stat per file) while still catching edits, additions and removals.
    """
    hasher = hashlib.sha256()
    hasher.update(f"v{CACHE_VERSION}|{sample_time}|{sample_rate}|".encode("utf-8"))
    for name in sorted(os.listdir(user_dir)):
        if not name.endswith(".csv"):
            continue
        stat = os.stat(user_dir / name)
        hasher.update(f"{name}|{stat.st_size}|{stat.st_mtime_ns}|".encode("utf-8"))
    return hasher.hexdigest()


def entry_path(user_dir: Path, sample_time: int, sample_rate: int) -> Path:
    """Readable-but-unique cache filename: dataset, user, resolution, signature."""
    signature = directory_signature(user_dir, sample_time, sample_rate)
    dataset_name = user_dir.parent.parent.name if len(user_dir.parents) >= 2 else "dataset"
    name = f"{_slug(dataset_name)}__{_slug(user_dir.name)}__{sample_time}s{sample_rate}hz__{signature[:16]}.pt"
    return cache_dir() / name


def load(path: Path) -> torch.Tensor | None:
    """Return the cached tensor, or None if absent or unreadable."""
    if not cache_enabled() or not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        samples = payload["samples"]
    except Exception:
        # A corrupt or partially written entry must never break a run; it will be
        # rebuilt and overwritten below.
        return None
    return samples if isinstance(samples, torch.Tensor) else None


def store(path: Path, samples: torch.Tensor) -> None:
    """Write a cache entry atomically. Failures are silently ignored."""
    if not cache_enabled():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash or a concurrent run never leaves a torn file.
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        torch.save({"version": CACHE_VERSION, "samples": samples}, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        pass

"""Enumerate every BOXRR-derived artefact on this machine, by construction.

WHY. The BOXRR-23 DUA's clause 15 lets Berkeley terminate, after which all copies must be
destroyed INCLUDING DERIVED ONES. On this node that is not one directory: `.cache/samples/`
writes a separate entry per user per (sample_time, sample_rate, channels) combination, so
every resolution anyone runs creates another set, and checkpoints trained on BOXRR are
plausibly derived copies too. A hand-maintained list is wrong the first time someone runs a
new resolution at 3am.

So this derives the list instead of recording it. Cache filenames are built as
`{dataset}__{user}__{time}s{rate}hz_{channels}__{sig}.pt` (model/sample_cache.py), so a
BOXRR entry is identifiable from its own name at any resolution, including resolutions that
did not exist when this was written. Checkpoints are recovered from the results shards,
whose rows record `data_dirs` by name.

THE FAILURE DIRECTION THAT MATTERS. An inventory that UNDER-reports is worse than none,
because it licenses a "destroyed" claim that is false. Two guards against that: the cache
directory is read from `sample_cache.cache_dir()` rather than hardcoded, so an
`XRSEC_SAMPLE_CACHE_DIR` relocation cannot hide a set (and the default is scanned too, in
case the variable is set now but was not when entries were written); and `--verify` plants
a decoy, confirms it is found, removes it, and confirms it is then absent - a guard checked
in both directions rather than trusted.

Usage:
    python boxrr_inventory.py                 # report
    python boxrr_inventory.py --verify        # prove detection works, then report
    python boxrr_inventory.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

#: Matched case-insensitively against dataset names, cache filenames and shard fields.
#: Kept broad on purpose - over-reporting costs a look, under-reporting costs a false
#: claim that the obligation was met.
BOXRR_TOKENS = ("boxrr", "beatsaber", "beat_saber", "xror")


def _looks_boxrr(text: str) -> bool:
    lowered = str(text).lower()
    return any(token in lowered for token in BOXRR_TOKENS)


def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def cache_dirs() -> list[Path]:
    """Every directory that could hold cache entries - the configured one AND the default.

    Reading only the configured directory would miss entries written before someone set
    XRSEC_SAMPLE_CACHE_DIR, which is the under-reporting direction.
    """
    found = []
    try:
        from model.sample_cache import cache_dir, DEFAULT_CACHE_DIR
        found.append(Path(cache_dir()))
        found.append(Path(DEFAULT_CACHE_DIR))
    except Exception:
        found.append(REPO_ROOT / ".cache" / "samples")
    override = os.environ.get("XRSEC_SAMPLE_CACHE_DIR")
    if override:
        found.append(Path(override))
    unique, seen = [], set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_cache_entries() -> list[dict]:
    """BOXRR cache entries at EVERY resolution, grouped by the resolution in the name."""
    out = []
    for directory in cache_dirs():
        if not directory.is_dir():
            continue
        for entry in sorted(directory.glob("*.pt")):
            if not _looks_boxrr(entry.name):
                continue
            match = re.search(r"__(\d+)s(\d+)hz_([A-Za-z0-9-]+)__", entry.name)
            out.append({
                "path": str(entry),
                "bytes": _size(entry),
                "resolution": f"{match.group(1)}s{match.group(2)}hz/{match.group(3)}" if match else "unparsed",
            })
    return out


def find_processed() -> list[dict]:
    out = []
    for root in (REPO_ROOT / "processed_datasets", REPO_ROOT / "raw_datasets"):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and _looks_boxrr(child.name):
                users = child / "users"
                out.append({"path": str(child), "bytes": _size(child),
                            "users": len(list(users.iterdir())) if users.is_dir() else None})
    return out


def find_checkpoints() -> list[dict]:
    """Checkpoints whose run recorded a BOXRR training or test directory.

    Weights are plausibly a derived copy; the coordinator has flagged that rather than
    settled it, so they are listed and left for a human to decide about.
    """
    out, seen = [], set()
    runs_dir = REPO_ROOT / "results" / "runs"
    rows = []
    if runs_dir.is_dir():
        for shard in sorted(runs_dir.glob("*.jsonl")):
            for line in shard.read_text(errors="replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append((shard.name, json.loads(line)))
                    except json.JSONDecodeError:
                        continue
    for shard_name, row in rows:
        if not _looks_boxrr(f"{row.get('data_dirs', '')}|{row.get('test_dirs', '')}|{row.get('dataset_tag', '')}"):
            continue
        checkpoint = row.get("checkpoint")
        if not checkpoint or checkpoint in seen:
            continue
        seen.add(checkpoint)
        absolute = REPO_ROOT / checkpoint
        out.append({"checkpoint": checkpoint, "shard": shard_name,
                    "present_on_this_machine": absolute.exists(),
                    "bytes": _size(absolute) if absolute.exists() else 0,
                    "run_id": row.get("run_id"), "sweep_id": row.get("sweep_id")})
    return out


def build_report() -> dict:
    cache = find_cache_entries()
    processed = find_processed()
    checkpoints = find_checkpoints()
    by_resolution: dict[str, dict] = {}
    for entry in cache:
        bucket = by_resolution.setdefault(entry["resolution"], {"entries": 0, "bytes": 0})
        bucket["entries"] += 1
        bucket["bytes"] += entry["bytes"]
    present = [c for c in checkpoints if c["present_on_this_machine"]]
    return {
        "repo_root": str(REPO_ROOT),
        "cache_dirs_scanned": [str(d) for d in cache_dirs()],
        "cache_entries": len(cache),
        "cache_bytes": sum(e["bytes"] for e in cache),
        "cache_by_resolution": by_resolution,
        "processed_or_raw": processed,
        "checkpoints_recorded": len(checkpoints),
        "checkpoints_present_here": len(present),
        "checkpoint_bytes_here": sum(c["bytes"] for c in present),
        "detail": {"cache": cache, "checkpoints": checkpoints},
    }


def verify() -> int:
    """Plant a decoy, confirm it is FOUND, remove it, confirm it is then ABSENT.

    A guard is only a guard if it does both. An inventory that always returned an empty
    list would pass a 'does it run' check and fail catastrophically at the only moment it
    is ever needed.
    """
    directory = cache_dirs()[0]
    directory.mkdir(parents=True, exist_ok=True)
    decoy = directory / "BOXRR-23-Dataset__inventory-selftest-user__99s99hz_full__deadbeefdeadbeef.pt"
    baseline = len(find_cache_entries())
    decoy.write_bytes(b"decoy for clause-15 inventory self-test")
    try:
        found = find_cache_entries()
        names = [Path(e["path"]).name for e in found]
        assert decoy.name in names, "DECOY NOT FOUND - the inventory under-reports and must not be trusted"
        planted = [e for e in found if Path(e["path"]).name == decoy.name][0]
        assert planted["resolution"] == "99s99hz/full", \
            f"resolution parsed as {planted['resolution']!r}, so a novel resolution would be mis-bucketed"
        assert len(found) == baseline + 1, f"expected {baseline + 1} entries, got {len(found)}"
        print(f"  decoy FOUND, resolution parsed as {planted['resolution']}")
    finally:
        decoy.unlink(missing_ok=True)
    after = find_cache_entries()
    assert decoy.name not in [Path(e["path"]).name for e in after], "decoy still reported after removal"
    assert len(after) == baseline, f"expected {baseline} entries after cleanup, got {len(after)}"
    print("  decoy ABSENT after removal, count back to baseline")
    print("VERIFY PASSES: the inventory detects a planted artefact and does not invent one.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.verify:
        print("=== verifying the inventory in both directions ===")
        verify()
        print()

    report = build_report()
    print("=== BOXRR-derived artefacts on this machine ===")
    print(f"repo:  {report['repo_root']}")
    for directory in report["cache_dirs_scanned"]:
        print(f"cache scanned: {directory}")
    print(f"\ncache entries: {report['cache_entries']}  ({report['cache_bytes'] / 2**30:.2f} GiB)")
    for resolution, bucket in sorted(report["cache_by_resolution"].items()):
        print(f"   {resolution:>18}  {bucket['entries']:6d} entries  {bucket['bytes'] / 2**30:8.2f} GiB")
    print(f"\nprocessed/raw directories: {len(report['processed_or_raw'])}")
    for entry in report["processed_or_raw"]:
        users = f", {entry['users']} users" if entry["users"] is not None else ""
        print(f"   {entry['path']}  ({entry['bytes'] / 2**30:.2f} GiB{users})")
    print(f"\ncheckpoints recorded in shards: {report['checkpoints_recorded']}"
          f"  present here: {report['checkpoints_present_here']}"
          f"  ({report['checkpoint_bytes_here'] / 2**30:.2f} GiB)")
    if report["cache_entries"] == 0 and not report["processed_or_raw"]:
        print("\nNo BOXRR-derived data on this machine. Nothing is owed under clause 15 yet.")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

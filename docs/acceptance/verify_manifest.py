"""
Verify a pulled corpus against a per-file `<size> <path>` manifest, file by file.

"The totals agree" and "every file agrees" are different claims: a truncated file
compensated by a longer one passes the first and fails the second, and that is the shape an
rsync failure actually takes. This diffs the two sets by path and size and prints every
difference, classified as missing / extra / size mismatch, with the totals last.

    python docs/acceptance/verify_manifest.py <manifest> <processed_datasets root> [--prefix CrossApplicationXR_Dataset]

Paths in the manifest are relative to the processed_datasets root (they start with the
dataset directory name). --prefix restricts the comparison to one dataset, so a corpus can
be verified as it arrives while others are still in flight. Sidecars the loader never reads
(root-level tasks.csv / users.csv) are compared like anything else; PROVENANCE.md and
CITATION.txt are excluded on both sides because they legitimately differ per machine.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

IGNORE_NAMES = {"PROVENANCE.md", "CITATION.txt"}


def read_manifest(path: pathlib.Path, prefix: str | None) -> dict[str, int]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        size, rel = line.split(" ", 1)
        rel = rel[2:] if rel.startswith("./") else rel      # a prefix, not a character class
        if prefix and not rel.startswith(prefix + "/"):
            continue
        if pathlib.Path(rel).name in IGNORE_NAMES:
            continue
        out[rel] = int(size)
    return out


def scan(root: pathlib.Path, prefix: str | None) -> dict[str, int]:
    out = {}
    base = root / prefix if prefix else root
    for dirpath, _, files in os.walk(base):
        for f in files:
            if f in IGNORE_NAMES:
                continue
            full = pathlib.Path(dirpath) / f
            rel = str(full.relative_to(root)).replace(os.sep, "/")
            if prefix and not rel.startswith(prefix + "/"):
                continue
            out[rel] = full.stat().st_size
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("root")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    want = read_manifest(pathlib.Path(args.manifest), args.prefix)
    have = scan(pathlib.Path(args.root), args.prefix)
    assert want, "manifest subset is empty - wrong prefix?"
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    mismatch = sorted(p for p in set(want) & set(have) if want[p] != have[p])
    for p in missing[:20]:
        print(f"MISSING   {p}")
    for p in extra[:20]:
        print(f"EXTRA     {p} ({have[p]} bytes)")
    for p in mismatch[:20]:
        print(f"SIZE      {p}: manifest {want[p]} local {have[p]}")
    ok = not (missing or extra or mismatch)
    print(f"manifest {len(want)} files / {sum(want.values()):,} bytes; local {len(have)} files / "
          f"{sum(have.values()):,} bytes; missing {len(missing)}, extra {len(extra)}, size mismatches {len(mismatch)} "
          f"-> {'VERIFIED' if ok else 'NOT VERIFIED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

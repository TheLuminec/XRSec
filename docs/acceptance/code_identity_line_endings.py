"""
Acceptance for the line-ending-normalised code identity (Coordinator, COORDINATION.md,
2026-09-06): after the change, the digest computed on the working tree must equal the digest
computed from git's stored blobs of the same commit. The pre-fix raw-byte digests of both are
printed as well, so rows recorded before the fix can be related to rows after it.

    python docs/acceptance/code_identity_line_endings.py [repo root] [commit]

With a commit, the stored-blob path is taken from that commit instead of HEAD (the working
tree is still hashed as it is), which relates the pre-fix code to its new identity.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
COMMIT = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
sys.path.insert(0, str(root / "model"))
from results_log import digest_tree  # noqa: E402


def raw_digest(tree: Path) -> str:
    """The pre-fix algorithm: raw bytes, so it follows the checkout's line endings."""
    digest = hashlib.sha1()
    for path in sorted(Path(tree).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(tree)).replace(chr(92), "/").encode("utf-8"))
        digest.update(hashlib.sha1(path.read_bytes()).digest())
    return digest.hexdigest()[:10]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


head = git("rev-parse", "--short", COMMIT).decode().strip()
files = [f for f in git("ls-tree", "-r", "--name-only", COMMIT, "model").decode().split() if f.endswith(".py")]
with tempfile.TemporaryDirectory() as tmp:
    blobs = Path(tmp) / "model"
    for f in files:
        out = blobs / Path(f).relative_to("model")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{COMMIT}:{f}"))          # stored blob bytes, LF
    work_norm, blob_norm = digest_tree(root / "model"), digest_tree(blobs)
    work_raw, blob_raw = raw_digest(root / "model"), raw_digest(blobs)

print(f"commit {head}: normalised digest, working tree {work_norm} | stored blobs {blob_norm}")
print(f"pre-fix raw-byte digests, for relating older rows: working tree {work_raw} | stored blobs {blob_raw}")
if COMMIT != "HEAD":
    print(f"(stored blobs of {head}; the working tree is HEAD, so the two need not agree)")
    sys.exit(0)
if work_norm != blob_norm:
    print("ACCEPTANCE FAILED: the working tree and the stored blobs disagree")
    sys.exit(1)
print("ACCEPTANCE PASSED: one identity for the tree however it was checked out")

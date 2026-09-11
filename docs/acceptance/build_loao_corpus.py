"""
Build the leave-one-application-out training copies of Across-XR (P3, Amendment 4).

The loader takes every CSV in a user directory and has no per-file exclusion, so "train on
four applications" needs a corpus in which the fifth application's sessions are absent.
For each held-out application X this writes

    processed_datasets/CrossApplicationXR_LOAO_<X>/users/<id>/<the four other sessions>.csv

as SYMLINKS to the real files (no data is copied; the source stays the single verified
corpus), plus splits.json and CITATION.txt, for all 49 users. Asserts on the result: 49 users,
4 sessions each, none of them X, every link resolving to a file in the verified corpus.

The sample cache keys on dataset name + CSV name/size/mtime, and symlinks stat through to
the real files, so the four kept sessions per user re-use nothing (different dataset name)
but build in seconds.
"""
import os, pathlib, sys

MAIN = pathlib.Path("/run/media/feng/Data/CalebProject/XRSec/processed_datasets")
SRC = MAIN / "CrossApplicationXR_Dataset"
GAMES = ("superhot_vr", "half_life_alyx", "beat_saber", "synth_riders", "social_vr")


def build(held_out: str) -> pathlib.Path:
    assert held_out in GAMES, held_out
    root = MAIN / f"CrossApplicationXR_LOAO_{held_out}"
    users = root / "users"
    users.mkdir(parents=True, exist_ok=True)
    for extra in ("splits.json",):
        target = root / extra
        if not target.exists():
            target.symlink_to(SRC / extra)
    if not (users / "CITATION.txt").exists():
        (users / "CITATION.txt").symlink_to(SRC / "users" / "CITATION.txt")
    count = 0
    for user_dir in sorted(p for p in (SRC / "users").iterdir() if p.is_dir()):
        out = users / user_dir.name
        out.mkdir(exist_ok=True)
        kept = []
        for csv in sorted(user_dir.glob("*.csv")):
            if csv.name.startswith(held_out):
                continue
            link = out / csv.name
            if not link.exists():
                link.symlink_to(csv)
            assert link.resolve() == csv.resolve() and link.stat().st_size > 0
            kept.append(csv.name)
        assert len(kept) == 4 and not any(k.startswith(held_out) for k in kept), (user_dir, kept)
        count += 1
    assert count == 49, count
    return root


if __name__ == "__main__":
    targets = sys.argv[1:] or list(GAMES)
    for g in targets:
        root = build(g)
        n_users = sum(1 for p in (root / "users").iterdir() if p.is_dir())
        n_csv = sum(1 for _ in (root / "users").rglob("*.csv"))
        print(f"{root.name}: {n_users} users, {n_csv} session links (held out: {g})")

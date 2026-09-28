"""
Build processed_datasets/Questset_g2: Questset group 2 only (Medal of Honor + Forklift Simulator, 30
people), as SYMLINKS to the verified Questset corpus, for the reverse-direction exposure arm
(questset_exposure_REGISTERED.md). Group 1 stays out of every training set and is scored as unseen.

A separate dataset name is the point, not a convenience: the normaliser keys statistics by dataset
name, so a run trained on Questset_g2 holds statistics under "Questset_g2", and scoring the full
"Questset" corpus later falls back to a target fit - the same convention as every comparator that
never saw Questset. Symlinks stat through to the real files, so the sample cache re-uses nothing
(different dataset name) and nothing is copied. Machine-local: run it on every node that trains or
scores, from the repo root.

    .venv/bin/python docs/acceptance/build_questset_subset.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "processed_datasets" / "Questset"
DST = ROOT / "processed_datasets" / "Questset_g2"
GAMES = ("forklift_simulator.csv", "medal_of_honor.csv")


def build() -> int:
    assert (SRC / "users").is_dir(), f"no Questset corpus at {SRC}"
    users = DST / "users"
    users.mkdir(parents=True, exist_ok=True)
    for extra in ("CITATION.txt", "manifest.json"):
        if (SRC / extra).exists() and not (DST / extra).exists():
            (DST / extra).symlink_to(SRC / extra)
    count = 0
    for user_dir in sorted(p for p in (SRC / "users").iterdir() if p.is_dir() and p.name.startswith("g2")):
        out = users / user_dir.name
        out.mkdir(exist_ok=True)
        csvs = sorted(c.name for c in user_dir.glob("*.csv"))
        assert tuple(csvs) == GAMES, (user_dir, csvs)
        for name in csvs:
            link = out / name
            if not link.exists():
                link.symlink_to(user_dir / name)
            assert link.resolve() == (user_dir / name).resolve() and link.stat().st_size > 0
        count += 1
    assert count == 30, count
    present = sorted(p.name for p in users.iterdir() if p.is_dir())
    assert len(present) == 30 and all(n.startswith("g2") for n in present), present
    assert not any(p.name.startswith("g1") for p in users.iterdir()), "a group-1 user is in the training corpus"
    return count


if __name__ == "__main__":
    n = build()
    n_csv = sum(1 for _ in (DST / "users").rglob("*.csv"))
    print(f"{DST.name}: {n} users, {n_csv} session links, group 1 absent")
    sys.exit(0)

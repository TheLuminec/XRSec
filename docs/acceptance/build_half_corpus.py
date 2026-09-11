"""
Build CrossApplicationXR_HALF for Amendment 5: every Across-XR session truncated to its
first half by row, as REAL files (a truncated file cannot be a symlink), for all 49 users.

Dose then halves at fixed identity count and fixed people, which is the only way to ask
whether the objective needed to see more of Across-XR separately from whether a bigger base
helped. Asserts on the result: 49 users, 5 sessions each, every output has the source
header and floor(n/2) data rows, SessionTime still starts at 0.
"""
import pathlib, sys

MAIN = pathlib.Path("/run/media/feng/Data/CalebProject/XRSec/processed_datasets")
SRC = MAIN / "CrossApplicationXR_Dataset"
DST = MAIN / "CrossApplicationXR_HALF"


def truncate(src: pathlib.Path, dst: pathlib.Path) -> tuple[int, int]:
    with src.open("r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    header, rows = lines[0], lines[1:]
    keep = rows[: len(rows) // 2]
    with dst.open("w", encoding="utf-8", newline="") as f:
        f.write(header)
        f.writelines(keep)
    assert keep and keep[0].split(",")[0] == "0.0", (src, keep[0][:40])
    return len(rows), len(keep)


if __name__ == "__main__":
    users = DST / "users"
    users.mkdir(parents=True, exist_ok=True)
    for extra in ("splits.json",):
        if not (DST / extra).exists():
            (DST / extra).symlink_to(SRC / extra)
    if not (users / "CITATION.txt").exists():
        (users / "CITATION.txt").symlink_to(SRC / "users" / "CITATION.txt")
    n_users, total_in, total_out = 0, 0, 0
    for user_dir in sorted(p for p in (SRC / "users").iterdir() if p.is_dir()):
        out = users / user_dir.name
        out.mkdir(exist_ok=True)
        csvs = sorted(user_dir.glob("*.csv"))
        assert len(csvs) == 5, user_dir
        for csv in csvs:
            a, b = truncate(csv, out / csv.name)
            total_in += a
            total_out += b
        n_users += 1
    assert n_users == 49
    print(f"{DST.name}: {n_users} users, {total_in:,} source rows -> {total_out:,} kept ({100 * total_out / total_in:.1f}%)")

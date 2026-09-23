"""Build the leave-script-out (LSO) corpora from the verified Nymeria corpus, deterministically, by
hard link (same bytes, same mtime, so the sample cache is shared; falls back to copy).

  Nymeria_LSO_train/users/<user>/  every recording whose script is NOT held out, all 236 users
  Nymeria_LSO_test/users/<user>/   every recording whose script IS held out, only the held-out users
                                   with >= 2 such recordings (positives are cross-script by construction)

Held-out scripts (registered rule: 5 scripts, <= 25 % of sequences, maximise held-out users with >= 2
held-out-script recordings): S10-Housekeeping, S13-Charades, S2-Where_is_X, S5-Workout, S6-Dance.
Writes docs/acceptance/nymeria_lso_manifest.txt (dst -> src, sha256) so another node rebuilds and
verifies the same trees from its own verified corpus.
"""
import csv, hashlib, os, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; PD = ROOT / "processed_datasets"
SRC = PD / "Nymeria_Dataset" / "users"
HELD_SCRIPTS = {"S10-Housekeeping", "S13-Charades", "S2-Where_is_X", "S5-Workout", "S6-Dance"}
scripts = {(r["participant"], r["act"]): r["script"] for r in csv.DictReader(open(ROOT / "docs/acceptance/nymeria_sequence_scripts.csv"))}
held_users = [l.strip() for l in open(ROOT / "docs/acceptance/nymeria_in_domain_heldout48.txt") if l.strip() and not l.startswith("#")]
def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)
def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
verify = "--verify" in sys.argv
lines, n_train, n_test, test_users = [], 0, 0, []
for user in sorted(d.name for d in SRC.iterdir() if d.is_dir()):
    files = sorted(f for f in os.listdir(SRC / user) if f.endswith(".csv"))
    held = [f for f in files if scripts[(user, f[:-4])] in HELD_SCRIPTS]
    train = [f for f in files if scripts[(user, f[:-4])] not in HELD_SCRIPTS]
    for f in train:
        dst = PD / "Nymeria_LSO_train" / "users" / user / f; link(SRC / user / f, dst); lines.append((dst, SRC / user / f)); n_train += 1
    if user in held_users and len(held) >= 2:
        test_users.append(user)
        for f in held:
            dst = PD / "Nymeria_LSO_test" / "users" / user / f; link(SRC / user / f, dst); lines.append((dst, SRC / user / f)); n_test += 1
for c in ("Nymeria_LSO_train", "Nymeria_LSO_test"):
    link(SRC / "CITATION.txt", PD / c / "users" / "CITATION.txt")
man = ROOT / "docs/acceptance/nymeria_lso_manifest.txt"
with open(man, "w") as f:
    f.write("# dst\tsrc\tsha256 (held-out scripts: %s)\n" % ", ".join(sorted(HELD_SCRIPTS)))
    for dst, src in lines:
        f.write(f"{dst.relative_to(PD)}\t{src.relative_to(PD)}\t{sha(dst) if verify else ''}\n")
tr_users = len([d for d in (PD / "Nymeria_LSO_train/users").iterdir() if d.is_dir()])
print(f"LSO_train: {tr_users} users, {n_train} files | LSO_test: {len(test_users)} users, {n_test} files | manifest {man.name} ({len(lines)} lines{', sha256' if verify else ''})")
print("test users:", " ".join(test_users))

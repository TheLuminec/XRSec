"""Produce both arms' user lists for the Nymeria in-domain arm (nymeria_in_domain_REGISTERED.md,
Amendment 3), per seed, from the pipeline's OWN select_validation_users on the directories the loader
will read - then assert the registered counts before writing anything.

    .venv/bin/python docs/acceptance/nymeria_in_domain_lists.py --seed 1 [--seed 2 ...] [--write]

Writes configs/nymeria_in_domain_{control,treatment}_s<seed>.yaml (Hydra configs composing on top of
config.yaml) so the lists never travel on a command line. Run it on the node that runs the arm; the
lists it prints must agree with AVALON's for the same seed, which is itself a corpus-agreement check.
"""
from __future__ import annotations
import argparse, hashlib, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from dataset import select_validation_users  # noqa: E402

PD = ROOT / "processed_datasets"
BOXRR, ALYX, NYM = PD / "BOXRR-23_Dataset" / "users", PD / "who_is_alyx" / "users", PD / "Nymeria_Dataset" / "users"
HELDOUT = ROOT / "docs" / "acceptance" / "nymeria_in_domain_heldout48.txt"
FIXED = dict(mode="train", experiment="nymeria_in_domain", encoding="dyn", sample_time=10, sample_rate=20,
             window_stride=5, extractor="bilstm", objective="identity_softmax", embedding_dim=128,
             normalize="per_dataset", within_dataset_negatives=True, cross_session_positives=True,
             epochs=120, early_stopping_patience=15, val_user_fraction=0.25, max_users=None,
             test_dirs=[], swap_data=False, test_on_excluded=True, channels="full", resample="nearest")

def users(d: Path) -> list[str]:
    return sorted(str(d / n) for n in os.listdir(d) if (d / n).is_dir())

def digest(paths) -> str:
    return hashlib.sha256("\n".join(Path(p).name for p in sorted(paths)).encode()).hexdigest()[:12]

def yaml_list(key, items):
    return f"{key}:\n" + "".join(f"  - {p}\n" for p in items) if items else f"{key}: []\n"

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    for d in (BOXRR, ALYX, NYM):
        assert d.is_dir(), f"missing corpus: {d}"
    held_names = [l.strip() for l in HELDOUT.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(held_names) == 48, len(held_names)
    held = [str(NYM / n) for n in held_names]
    for h in held: assert Path(h).is_dir(), f"held-out user missing on this machine: {h}"
    b, al, ny = users(BOXRR), users(ALYX), users(NYM)
    print(f"corpus: BOXRR {len(b)}  alyx {len(al)}  Nymeria {len(ny)}  (Nymeria list sha {digest(ny)})")
    nym_rest = [u for u in ny if u not in set(held)]
    data_dirs = [str(BOXRR), str(ALYX), str(NYM)]
    for seed in a.seed:
        # CONTROL: drop every non-held-out Nymeria user, then the pipeline's own 25% draw over BOXRR+alyx
        V = select_validation_users(data_dirs, held + nym_rest, 0.25, seed)
        assert all(not v.startswith(str(NYM)) for v in V)
        ctrl_train = [u for u in b + al if u not in set(V)]
        # TREATMENT: same V pinned; Nymeria drawn by the pipeline (47 of 188); drop the LAST 141 BOXRR training users
        nym_val = select_validation_users(data_dirs, held, 0.25, seed, explicit=V)
        nym_val_only = [v for v in nym_val if v.startswith(str(NYM))]
        n_nym_train = len(nym_rest) - len(nym_val_only)
        b_train_ctrl = [u for u in b if u not in set(V)]
        drop_b = b_train_ctrl[-n_nym_train:]
        treat_train = [u for u in b_train_ctrl if u not in set(drop_b)] + [u for u in al if u not in set(V)] \
                      + [u for u in nym_rest if u not in set(nym_val_only)]
        print(f"seed {seed}: V={len(V)} (BOXRR {sum(v.startswith(str(BOXRR)) for v in V)}, alyx {sum(v.startswith(str(ALYX)) for v in V)}) "
              f"| Nymeria val {len(nym_val_only)} train {n_nym_train} | control train {len(ctrl_train)} | treatment train {len(treat_train)} "
              f"| BOXRR dropped {len(drop_b)} | nested: {set(u for u in treat_train if u.startswith(str(BOXRR))) <= set(b_train_ctrl)} "
              f"| V sha {digest(V)} drop sha {digest(drop_b)}")
        assert len(ctrl_train) == len(treat_train), "arms are not identity-matched"
        assert set(u for u in treat_train if u.startswith(str(BOXRR))) <= set(b_train_ctrl)
        if a.write:
            for arm, drop in (("control", nym_rest), ("treatment", drop_b)):
                out = ROOT / "configs" / f"nymeria_in_domain_{arm}_s{seed}.yaml"
                body = "defaults:\n  - config\n  - _self_\n\n" + f"seed: {seed}\n"
                for k, v in FIXED.items():
                    body += f"{k}: {'null' if v is None else (str(v).lower() if isinstance(v, bool) else v)}\n" if not isinstance(v, list) else yaml_list(k, v)
                body += yaml_list("data_dirs", data_dirs) + yaml_list("exclude_users", held) + yaml_list("validation_users", V) + yaml_list("drop_users", drop)
                out.write_text(body); print(f"  wrote {out.relative_to(ROOT)} ({len(drop)} drop_users, {len(V)} validation_users)")
    return 0

if __name__ == "__main__":
    sys.exit(main())

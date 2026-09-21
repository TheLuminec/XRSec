"""Produce both arms' user lists for the Nymeria in-domain arm (nymeria_in_domain_REGISTERED.md,
Amendment 3), per seed, from the pipeline's OWN select_validation_users - then assert the registered
counts and the held-out users' absence from every training list before writing anything.

    # reference node (AVALON, once): draw, assert, write machine-independent reference lists + local configs
    .venv/bin/python docs/acceptance/nymeria_in_domain_lists.py --seed 1 --seed 2 --seed 3 --write
    # any other node: rebuild local configs FROM the committed reference (no draw), assert every name exists
    .venv313/bin/python docs/acceptance/nymeria_in_domain_lists.py --seed 1 --from-reference

Reference lists are user directory NAMES (docs/acceptance/nymeria_in_domain_lists_s<seed>.json), so
they do not depend on machine paths or on numpy's Generator stream (Miami's note 1: the treatment's 47
Nymeria validation users are a draw, and a draw is not stable across numpy feature releases). The
configs (configs/nymeria_in_domain_{control,treatment}_s<seed>.yaml, gitignored, absolute paths) compose
on config.yaml so 1,024 paths never travel on a command line. Every list in the treatment config is
explicit, so the pipeline draws nothing at runtime on either arm.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from dataset import select_validation_users  # noqa: E402

PD = ROOT / "processed_datasets"
CORPORA = {"BOXRR-23_Dataset": PD / "BOXRR-23_Dataset" / "users", "who_is_alyx": PD / "who_is_alyx" / "users",
           "Nymeria_Dataset": PD / "Nymeria_Dataset" / "users"}
BOXRR, ALYX, NYM = CORPORA["BOXRR-23_Dataset"], CORPORA["who_is_alyx"], CORPORA["Nymeria_Dataset"]
HELDOUT = ROOT / "docs" / "acceptance" / "nymeria_in_domain_heldout48.txt"
REF = lambda seed: ROOT / "docs" / "acceptance" / f"nymeria_in_domain_lists_s{seed}.json"
FIXED = dict(mode="train", experiment_name="nymeria_in_domain",  # the logger records experiment_name; `experiment` is inert encoding="dyn", sample_time=10, sample_rate=20,
             window_stride=5, extractor="bilstm", objective="identity_softmax", embedding_dim=128,
             normalize="per_dataset", within_dataset_negatives=True, cross_session_positives=True,
             epochs=120, early_stopping_patience=15, val_user_fraction=0.25, max_users=None,
             test_dirs=[], swap_data=False, test_on_excluded=True, channels="full", resample="nearest")

def users(d: Path) -> list[str]:
    return sorted(str(d / n) for n in os.listdir(d) if (d / n).is_dir())

def digest(paths) -> str:  # over NAMES, sorted: machine-independent
    return hashlib.sha256("\n".join(sorted(Path(p).name for p in paths)).encode()).hexdigest()[:12]

def names(paths) -> list[str]:  # "<corpus>/<user>" so a name is unambiguous across corpora
    return sorted(f"{Path(p).parent.parent.name}/{Path(p).name}" for p in paths)

def paths_of(items) -> list[str]:
    out = []
    for it in items:
        corpus, user = it.split("/", 1); p = CORPORA[corpus] / user
        assert p.is_dir(), f"reference names a user directory absent on this machine: {p}"
        out.append(str(p))
    return sorted(out)

def q(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"

def yaml_list(key, items):
    return f"{key}:\n" + "".join(f"  - {q(p)}\n" for p in items) if items else f"{key}: []\n"

def write_config(arm, seed, data_dirs, held, val, drop):
    out = ROOT / "configs" / f"nymeria_in_domain_{arm}_s{seed}.yaml"
    body = "defaults:\n  - config\n  - _self_\n\n" + f"seed: {seed}\n"
    for k, v in FIXED.items():
        body += yaml_list(k, v) if isinstance(v, list) else f"{k}: {'null' if v is None else (str(v).lower() if isinstance(v, bool) else (q(v) if isinstance(v, str) else v))}\n"
    body += yaml_list("data_dirs", data_dirs) + yaml_list("exclude_users", held) + yaml_list("validation_users", val) + yaml_list("drop_users", drop)
    out.write_text(body); return out

def check_arm(arm, all_users, held, val, drop):
    """The registered properties, asserted on the lists the config will carry - never inferred."""
    H, V, D = set(held), set(val), set(drop)
    train = [u for u in all_users if u not in H and u not in V and u not in D]
    assert not (H & V) and not (H & D), f"{arm}: a held-out user appears in validation or drop"
    assert not (set(train) & H), f"{arm}: a held-out user would be TRAINED on"
    assert not (V & D), f"{arm}: validation and drop overlap"
    return train

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--write", action="store_true", help="draw from the pipeline, write reference + configs (reference node)")
    ap.add_argument("--from-reference", action="store_true", help="rebuild configs from the committed reference lists (no draw)")
    a = ap.parse_args()
    for d in CORPORA.values(): assert d.is_dir(), f"missing corpus: {d}"
    held_names = [l.strip() for l in HELDOUT.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(held_names) == 48, len(held_names)
    held = paths_of(f"Nymeria_Dataset/{n}" for n in held_names)
    b, al, ny = users(BOXRR), users(ALYX), users(NYM)
    all_users = b + al + ny
    print(f"corpus: BOXRR {len(b)}  alyx {len(al)}  Nymeria {len(ny)}  (Nymeria list sha {digest(ny)})")
    nym_rest = [u for u in ny if u not in set(held)]
    data_dirs = [str(BOXRR), str(ALYX), str(NYM)]
    for seed in a.seed:
        if a.from_reference:
            ref = json.loads(REF(seed).read_text())
            V, nym_val, drop_b = paths_of(ref["validation_control"]), paths_of(ref["validation_treatment_nymeria"]), paths_of(ref["drop_treatment_boxrr"])
            assert paths_of(ref["held_out"]) == held and paths_of(ref["drop_control_nymeria"]) == sorted(nym_rest)
            src = "reference"
        else:
            V = select_validation_users(data_dirs, held + nym_rest, 0.25, seed)                 # control's draw: BOXRR+alyx only
            assert all(not v.startswith(str(NYM)) for v in V)
            nym_val = [v for v in select_validation_users(data_dirs, held, 0.25, seed, explicit=V) if v.startswith(str(NYM))]
            b_train_ctrl = [u for u in b if u not in set(V)]
            drop_b = b_train_ctrl[-(len(nym_rest) - len(nym_val)):]                            # last 141 BOXRR training users
            src = "draw"
        val_treat = sorted(V + nym_val)
        ctrl_train = check_arm("control", all_users, held, V, nym_rest)
        treat_train = check_arm("treatment", all_users, held, val_treat, drop_b)
        b_ctrl, b_treat = {u for u in ctrl_train if u.startswith(str(BOXRR))}, {u for u in treat_train if u.startswith(str(BOXRR))}
        assert len(ctrl_train) == len(treat_train), "arms are not identity-matched"
        assert b_treat <= b_ctrl, "treatment BOXRR training users are not nested in the control's"
        assert {u for u in ctrl_train if u.startswith(str(ALYX))} == {u for u in treat_train if u.startswith(str(ALYX))}
        print(f"seed {seed} [{src}]: V={len(V)} (BOXRR {sum(v.startswith(str(BOXRR)) for v in V)}, alyx {sum(v.startswith(str(ALYX)) for v in V)}) "
              f"| Nymeria val {len(nym_val)} train {len(nym_rest) - len(nym_val)} | control train {len(ctrl_train)} | treatment train {len(treat_train)} "
              f"| BOXRR dropped {len(drop_b)} | nested True | held-out absent from both: True")
        print(f"        digests: V {digest(V)}  Vtreat {digest(val_treat)}  dropB {digest(drop_b)}  dropC {digest(nym_rest)}  held {digest(held)}")
        if a.write or a.from_reference:
            if a.write:
                REF(seed).write_text(json.dumps({"seed": seed, "held_out": names(held), "validation_control": names(V),
                    "validation_treatment_nymeria": names(nym_val), "drop_treatment_boxrr": names(drop_b),
                    "drop_control_nymeria": names(nym_rest), "counts": {"control_train": len(ctrl_train), "treatment_train": len(treat_train)},
                    "digests": {"V": digest(V), "Vtreat": digest(val_treat), "dropB": digest(drop_b), "dropC": digest(nym_rest), "held": digest(held), "nymeria_users": digest(ny)}}, indent=1))
                print(f"  wrote {REF(seed).relative_to(ROOT)}")
            for arm, val, drop in (("control", V, nym_rest), ("treatment", val_treat, drop_b)):
                out = write_config(arm, seed, data_dirs, held, val, drop)
                print(f"  wrote {out.relative_to(ROOT)} ({len(val)} validation_users, {len(drop)} drop_users, {len(held)} exclude_users)")
    return 0

if __name__ == "__main__":
    sys.exit(main())

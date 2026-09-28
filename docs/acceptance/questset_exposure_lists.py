"""Configs for the reverse-direction exposure arm (questset_exposure_REGISTERED.md).

The Nymeria in-domain treatment, seed-matched, with Questset group 2 (30 people x Medal of Honor +
Forklift Simulator) swapped in for the last 30 BOXRR training users post-draw. Validation users and
the evaluation set (the 48 held-out Nymeria users) are the treatment's own, so the run selects its
epoch on the same people and reproduces the treatment's evaluation population exactly. Training
identities stay 3,072.

    .venv/bin/python docs/acceptance/build_questset_subset.py          # first, on this machine
    .venv/bin/python docs/acceptance/questset_exposure_lists.py --seed 1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
import yaml
from nymeria_in_domain_lists import FIXED, CORPORA, PD, digest, q, yaml_list, users  # noqa: E402
from exposure_breadth_lists import paths_of  # noqa: E402

N_SWAP = 30


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    ref = json.loads((ROOT / "docs/acceptance" / f"nymeria_in_domain_lists_s{a.seed}.json").read_text())
    held = paths_of(ref["held_out"]); V = paths_of(ref["validation_control"])
    nym_val = paths_of(ref["validation_treatment_nymeria"]); dropB = paths_of(ref["drop_treatment_boxrr"])
    b_train = [u for u in users(CORPORA["BOXRR-23_Dataset"]) if u not in set(V) and u not in set(dropB)]
    drop_more = b_train[-N_SWAP:]                                  # last 30 in sorted order, post-draw
    qs = PD / "Questset_g2" / "users"; assert qs.is_dir(), f"build_questset_subset.py first: {qs}"
    qs_users = users(qs); assert len(qs_users) == N_SWAP and all(Path(u).name.startswith("g2") for u in qs_users)
    fixed = dict(FIXED, experiment_name="questset_exposure_g2")
    out = ROOT / "configs" / f"questset_exposure_g2_s{a.seed}.yaml"
    body = "defaults:\n  - config\n  - _self_\n\n" + f"seed: {a.seed}\n"
    for k, v in fixed.items():
        body += yaml_list(k, v) if isinstance(v, list) else f"{k}: {'null' if v is None else (str(v).lower() if isinstance(v, bool) else (q(v) if isinstance(v, str) else v))}\n"
    data_dirs = [str(CORPORA["BOXRR-23_Dataset"]), str(CORPORA["who_is_alyx"]), str(CORPORA["Nymeria_Dataset"]), str(qs)]
    body += yaml_list("data_dirs", data_dirs) + yaml_list("exclude_users", sorted(held)) \
          + yaml_list("validation_users", sorted(V + nym_val)) + yaml_list("drop_users", sorted(dropB + drop_more))
    out.write_text(body); w = yaml.safe_load(out.read_text())
    for k, v in fixed.items(): assert w[k] == v, (k, w[k], v)          # parse the artefact back, every key
    assert w["seed"] == a.seed and w["data_dirs"] == data_dirs
    assert len(w["exclude_users"]) == 48 and len(w["validation_users"]) == 1071 and len(w["drop_users"]) == 141 + N_SWAP
    assert not set(w["exclude_users"]) & set(w["validation_users"]) and not set(w["drop_users"]) & set(w["validation_users"])
    alyx_train = [u for u in users(CORPORA["who_is_alyx"]) if u not in set(V)]
    n_train = len(b_train) - len(drop_more) + len(alyx_train) + (236 - 48 - len(nym_val)) + len(qs_users)
    assert n_train == 3072, n_train
    print(f"wrote {out.relative_to(ROOT)} | training identities {n_train} (BOXRR {len(b_train)-len(drop_more)}, "
          f"alyx {len(alyx_train)}, Nymeria {236 - 48 - len(nym_val)}, Questset g2 {len(qs_users)}) | "
          f"dropB+30 {digest(dropB + drop_more)} | excl {digest(held)} | val {digest(V + nym_val)}")


if __name__ == "__main__":
    main()

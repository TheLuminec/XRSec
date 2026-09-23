"""Configs for the Nymeria leave-script-out arm (nymeria_lso_REGISTERED.md) from the SAME reference
lists as the in-domain arm (nymeria_in_domain_lists_s<seed>.json): the 48 held-out people, the pinned
validation draw, the 141 BOXRR users dropped so identity count matches. Only the Nymeria root changes
(Nymeria_LSO_train: training scripts only) and the evaluation set is Nymeria_LSO_test (held-out
scripts, held-out people, >= 2 held-out-script recordings) via test_dirs.

    .venv/bin/python docs/acceptance/nymeria_lso_lists.py --seed 1 [--seed 2 ...]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
import yaml
from nymeria_in_domain_lists import FIXED, CORPORA, PD, digest, q, yaml_list  # noqa: E402

LSO_TRAIN = PD / "Nymeria_LSO_train" / "users"; LSO_TEST = PD / "Nymeria_LSO_test" / "users"
C = dict(CORPORA); C["Nymeria_Dataset"] = LSO_TRAIN     # every Nymeria name resolves under the LSO training tree

def paths_of(items):
    out = []
    for it in items:
        corpus, user = it.split("/", 1); p = C[corpus] / user
        assert p.is_dir(), f"missing on this machine: {p}"
        out.append(str(p))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, action="append", required=True); a = ap.parse_args()
    for d in list(C.values()) + [LSO_TEST]: assert d.is_dir(), d
    test_users = sorted(p.name for p in LSO_TEST.iterdir() if p.is_dir())
    for seed in a.seed:
        ref = json.loads((ROOT / "docs/acceptance" / f"nymeria_in_domain_lists_s{seed}.json").read_text())
        held = paths_of(ref["held_out"]); V = paths_of(ref["validation_control"]) + paths_of(ref["validation_treatment_nymeria"]); dropB = paths_of(ref["drop_treatment_boxrr"])
        assert set(test_users) <= {Path(h).name for h in held}, "LSO_test holds a user not in the held-out 48"
        fixed = dict(FIXED, experiment_name="nymeria_lso", test_on_excluded=False, test_dirs=[str(LSO_TEST)])
        out = ROOT / "configs" / f"nymeria_lso_s{seed}.yaml"
        body = "defaults:\n  - config\n  - _self_\n\n" + f"seed: {seed}\n"
        for k, v in fixed.items():
            body += yaml_list(k, v) if isinstance(v, list) else f"{k}: {'null' if v is None else (str(v).lower() if isinstance(v, bool) else (q(v) if isinstance(v, str) else v))}\n"
        body += yaml_list("data_dirs", [str(C["BOXRR-23_Dataset"]), str(C["who_is_alyx"]), str(LSO_TRAIN)]) + yaml_list("exclude_users", held) + yaml_list("validation_users", sorted(V)) + yaml_list("drop_users", dropB)
        out.write_text(body); w = yaml.safe_load(out.read_text())
        for k, v in fixed.items(): assert w[k] == v, (k, w[k], v)
        assert len(w["exclude_users"]) == 48 and len(w["validation_users"]) == 1071 and len(w["drop_users"]) == 141 and len(w["test_dirs"]) == 1
        print(f"seed {seed}: wrote {out.relative_to(ROOT)} | held {digest(held)} V {digest(V)} dropB {digest(dropB)} | test users {len(test_users)} (LSO_test) | Nymeria training users {236 - 48 - 47}")
if __name__ == "__main__": main()

"""Configs for the short-window arms (ballthrowing_cross_day_REGISTERED.md; broad_2s_REGISTERED.md).

The Nymeria in-domain treatment, seed-matched and unchanged in composition (3,072 training identities,
the treatment's own 1,071 validation users, the same 48 held-out Nymeria users), trained on SHORT
windows so that a ~3 s ball throw yields a window: sample_time=2, window_stride=5 - the treatment's
stride, so the number of training windows and the epoch cost stay close to the 10 s arm's.

    .venv/bin/python docs/acceptance/treatment_short_lists.py --seed 1 [--sample-time 2] [--arm control] [--encoding raw]

--arm control is the Nymeria control composition (every non-held-out Nymeria user dropped, the control's own
1,024 validation users, 3,072 BOXRR+alyx identities). --encoding changes only the encoding; the name gains
the encoding as a suffix when it is not dyn, so the dyn treatment's name and config are unchanged.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "acceptance")); sys.path.insert(0, str(ROOT / "model"))
import yaml
from nymeria_in_domain_lists import FIXED, CORPORA, digest, q, yaml_list, users  # noqa: E402
from exposure_breadth_lists import paths_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample-time", type=int, default=2)
    ap.add_argument("--arm", choices=("treatment", "control"), default="treatment")
    ap.add_argument("--encoding", choices=("dyn", "raw", "br"), default="dyn")
    a = ap.parse_args()
    ref = json.loads((ROOT / "docs/acceptance" / f"nymeria_in_domain_lists_s{a.seed}.json").read_text())
    held = paths_of(ref["held_out"]); V = paths_of(ref["validation_control"])
    nym_val = paths_of(ref["validation_treatment_nymeria"]); dropB = paths_of(ref["drop_treatment_boxrr"])
    if a.arm == "treatment":
        val, drop, n_drop = V + nym_val, dropB, 141
    else:
        val, drop, n_drop = V, paths_of(ref["drop_control_nymeria"]), 188
    name = f"{a.arm}_{a.sample_time}s" + ("" if a.encoding == "dyn" else f"_{a.encoding}")
    fixed = dict(FIXED, experiment_name=name, sample_time=a.sample_time, window_stride=5, encoding=a.encoding)
    out = ROOT / "configs" / f"{name}_s{a.seed}.yaml"
    body = "defaults:\n  - config\n  - _self_\n\n" + f"seed: {a.seed}\n"
    for k, v in fixed.items():
        body += yaml_list(k, v) if isinstance(v, list) else f"{k}: {'null' if v is None else (str(v).lower() if isinstance(v, bool) else (q(v) if isinstance(v, str) else v))}\n"
    data_dirs = [str(CORPORA["BOXRR-23_Dataset"]), str(CORPORA["who_is_alyx"]), str(CORPORA["Nymeria_Dataset"])]
    body += yaml_list("data_dirs", data_dirs) + yaml_list("exclude_users", sorted(held)) \
          + yaml_list("validation_users", sorted(val)) + yaml_list("drop_users", sorted(drop))
    out.write_text(body); w = yaml.safe_load(out.read_text())
    for k, v in fixed.items(): assert w[k] == v, (k, w[k], v)          # parse the artefact back, every key
    assert w["seed"] == a.seed and w["data_dirs"] == data_dirs
    assert len(w["exclude_users"]) == 48 and len(w["validation_users"]) == len(val) and len(w["drop_users"]) == n_drop
    assert len(val) == (1071 if a.arm == "treatment" else 1024)
    # resolve the split through the LOADER's own function (Q2 Amendment 1): nothing drawn, 3,072 trained
    from dataset import select_validation_users
    resolved = select_validation_users(w["data_dirs"], list(w["exclude_users"]) + list(w["drop_users"]),
                                       w["val_user_fraction"], a.seed, explicit=w["validation_users"])
    assert sorted(resolved) == sorted(val), f"loader draws {len(resolved)} validation users, not {len(val)}"
    kept = [u for d in w["data_dirs"] for u in users(Path(d))
            if u not in set(w["exclude_users"]) | set(w["drop_users"]) | set(resolved)]
    assert len(kept) == 3072, f"loader would train on {len(kept)} identities"
    print(f"wrote {out.relative_to(ROOT)} | {a.arm} {a.encoding} | sample_time {a.sample_time} stride 5 | training identities {len(kept)} | "
          f"drop {digest(drop)} | excl {digest(held)} | val {digest(val)}")


if __name__ == "__main__":
    main()

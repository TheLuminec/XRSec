"""Configs for experiment 2 (exposure_breadth_REGISTERED.md): one per held-out Across-XR application.
Reuses the in-domain arm's reference lists, drops 23 more BOXRR training users post-draw for the 23
Across-XR training users, pins Across-XR 23-31 as that corpus's explicit validation users, excludes and
evaluates Across-XR 32-48 beside the 48 Nymeria held-out.

    .venv/bin/python docs/acceptance/exposure_breadth_lists.py --seed 1 [--held-out beat_saber ...]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "docs" / "acceptance"))
import yaml
from nymeria_in_domain_lists import FIXED, CORPORA, PD, digest, q, yaml_list, users  # noqa: E402
GAMES = ("superhot_vr", "half_life_alyx", "beat_saber", "synth_riders", "social_vr")
AXR_TRAIN, AXR_VAL, AXR_TEST = range(0, 23), range(23, 32), range(32, 49)

def paths_of(items):
    out = []
    for it in items:
        c, u = it.split("/", 1); p = CORPORA[c] / u; assert p.is_dir(), p; out.append(str(p))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--held-out", action="append")
    a = ap.parse_args(); games = a.held_out or list(GAMES)
    ref = json.loads((ROOT / "docs/acceptance" / f"nymeria_in_domain_lists_s{a.seed}.json").read_text())
    held = paths_of(ref["held_out"]); V = paths_of(ref["validation_control"]); nym_val = paths_of(ref["validation_treatment_nymeria"]); dropB = paths_of(ref["drop_treatment_boxrr"])
    b_train = [u for u in users(CORPORA["BOXRR-23_Dataset"]) if u not in set(V) and u not in set(dropB)]
    drop_more = b_train[-len(AXR_TRAIN):]                      # 23 more, last in sorted order, post-draw
    for g in games:
        axr = PD / f"CrossApplicationXR_LOAO_{g}" / "users"; assert axr.is_dir(), f"build_loao_corpus.py first: {axr}"
        ax = lambda ids: sorted(str(axr / str(i)) for i in ids)
        for p in ax(AXR_TRAIN) + ax(AXR_VAL) + ax(AXR_TEST): assert Path(p).is_dir(), p
        fixed = dict(FIXED, experiment_name=f"exposure_breadth_{g}")
        out = ROOT / "configs" / f"exposure_breadth_{g}_s{a.seed}.yaml"
        body = "defaults:\n  - config\n  - _self_\n\n" + f"seed: {a.seed}\n"
        for k, v in fixed.items():
            body += yaml_list(k, v) if isinstance(v, list) else f"{k}: {'null' if v is None else (str(v).lower() if isinstance(v, bool) else (q(v) if isinstance(v, str) else v))}\n"
        data_dirs = [str(CORPORA["BOXRR-23_Dataset"]), str(CORPORA["who_is_alyx"]), str(CORPORA["Nymeria_Dataset"]), str(axr)]
        body += yaml_list("data_dirs", data_dirs) + yaml_list("exclude_users", sorted(held + ax(AXR_TEST))) \
              + yaml_list("validation_users", sorted(V + nym_val + ax(AXR_VAL))) + yaml_list("drop_users", sorted(dropB + drop_more))
        out.write_text(body); w = yaml.safe_load(out.read_text())
        for k, v in fixed.items(): assert w[k] == v, (k, w[k], v)
        n_train = len(b_train) - len(drop_more) + len([u for u in users(CORPORA["who_is_alyx"]) if u not in set(V)]) \
                  + (236 - 48 - len(nym_val)) + len(AXR_TRAIN)
        assert len(w["exclude_users"]) == 48 + 17 and len(w["validation_users"]) == 1071 + 9 and len(w["drop_users"]) == 141 + 23
        print(f"{g:16s} wrote {out.relative_to(ROOT)} | training identities {n_train} (BOXRR {len(b_train)-len(drop_more)}, alyx {len([u for u in users(CORPORA["who_is_alyx"]) if u not in set(V)])}, Nymeria {236 - 48 - len(nym_val)}, Across-XR 23) | dropB+23 {digest(dropB + drop_more)} | excl {digest(held + ax(AXR_TEST))}")

if __name__ == "__main__": main()

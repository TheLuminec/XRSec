"""
The C2-hi / Z-676 pair, exact by construction (Amendment 1 of across_xr_alignment_REGISTERED).

Capping BOXRR at 577 in one arm and 600 in the other equalises the identity count only
before the validation draw: `val_user_fraction=0.25` draws over each arm's OWN BOXRR+alyx
pool (653 against 676 users), so the two arms would validate on different people and train
on 513 against 507 identities, and the BOXRR training sets would not be nested even though
the subsamples are. This script makes the pair exact instead:

  Z-676   BOXRR seeded subsample of 600 + alyx 76; validation = the pipeline's own 25% draw
          over that pool, made EXPLICIT here and passed to both arms
  C2-hi   the same 600 + 76 and the same validation list, minus the LAST 23 BOXRR training
          users in the subsample's permutation order (so its BOXRR training users are a strict
          subset of Z-676's), plus Across-XR 0-22 in training and 23-31 in validation

Trained identities are then equal by construction, the validation people are identical, and
the only difference inside the pair is which 23 identities did which activity - the Nymeria
arm-B design. Writes docs/acceptance/c2_pair_users_seed<N>.json for the launcher, asserts the
properties on the lists the LOADERS hold, and measures C2-hi's Across-XR window dose.

    python docs/acceptance/c2_pair_lists.py <repo root> <seed>
"""
import contextlib, io, json, os, pathlib, sys, time
ROOT = sys.argv[1]
SEED = int(sys.argv[2])
MAIN = "/run/media/feng/Data/CalebProject/XRSec"
sys.path.insert(0, os.path.join(ROOT, "model"))
os.environ["XRSEC_SAMPLE_CACHE_DIR"] = MAIN + "/.cache/samples"
import numpy as np
import torch
from dataset import create_dataloader_from_path, select_user_subset, select_validation_users, _user_dirs_of, _seed_value

ALYX = MAIN + "/processed_datasets/who_is_alyx/users"
BOXRR = MAIN + "/processed_datasets/BOXRR-23_Dataset/users"
XR = MAIN + "/processed_datasets/CrossApplicationXR_Dataset/users"
CAP = {"BOXRR-23_Dataset": 600}
SWAP = 23

# Z-676's users and the validation draw the pipeline itself would make for them.
z_users = select_user_subset([BOXRR, ALYX], CAP, SEED)
z_val = [u for u in select_validation_users([BOXRR, ALYX], [], 0.25, SEED) if u in set(z_users)]
z_train_boxrr = [u for u in z_users if u.startswith(BOXRR) and u not in set(z_val)]
# The subsample is a prefix of one seeded permutation; drop the last 23 TRAINING users in
# that order so what remains is a strict subset and the arm is reproducible from the seed.
boxrr_all = [os.path.join(BOXRR, n) for n in sorted(os.listdir(BOXRR)) if os.path.isdir(os.path.join(BOXRR, n))]
rng = np.random.default_rng(_seed_value(SEED, 31))
order = [boxrr_all[i] for i in rng.permutation(len(boxrr_all))[:600]]
assert set(order) == {u for u in z_users if u.startswith(BOXRR)}, "permutation reconstruction differs from select_user_subset"
dropped = [u for u in order if u in set(z_train_boxrr)][-SWAP:]
xr_test = [f"{XR}/{u}" for u in range(32, 49)]
xr_val = [f"{XR}/{u}" for u in range(23, 32)]
lists = {"seed": SEED, "boxrr_cap": 600, "z676_validation_users": z_val,
         "c2hi_dropped_boxrr_train_users": dropped, "xr_validation_users": xr_val, "xr_test_users": xr_test}
out = pathlib.Path(ROOT) / "docs" / "acceptance" / f"c2_pair_users_seed{SEED}.json"
out.write_text(json.dumps(lists, indent=1), encoding="utf-8")
print(f"seed {SEED}: Z-676 users {len(z_users)}, validation {len(z_val)}, BOXRR train {len(z_train_boxrr)}, dropped {len(dropped)} -> {out.name}")

common = dict(sample_time=10, sample_rate=20, samples_per_user=64, channels="full", window_stride=5,
              encoding="dyn", normalize="per_dataset", within_dataset_negatives=True,
              cross_session_positives=True, val_user_fraction=0.0)
dev = torch.device("cpu")


def build(dirs, exclude, val, drop=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return create_dataloader_from_path(dirs, 256, dev, is_train=True, test_dir=[XR], exclude_users=exclude,
                                           swap_data=False, test_on_excluded=True, seed=SEED, return_val=True,
                                           validation_users=val, max_users=CAP, drop_users=drop, **common)


t0 = time.time()
z_tr, z_va, z_te = build([BOXRR, ALYX], xr_test, z_val)
# Across-XR 23-31 are DROPPED from C2-hi - neither trained on, validated on nor evaluated - so
# both arms choose their epoch on the identical 181 people and no target-corpus user is in
# C2-hi's selection signal (Coordinator, 2026-09-10).
h_tr, h_va, h_te = build([BOXRR, ALYX, XR], xr_test + dropped, z_val, drop=xr_val)
zt, zv, ht, hv = (_user_dirs_of(d) for d in (z_tr.dataset, z_va.dataset, h_tr.dataset, h_va.dataset))
z_b = {u for u in zt if u.startswith(BOXRR)}; h_b = {u for u in ht if u.startswith(BOXRR)}
h_x = sorted(int(os.path.basename(u)) for u in ht if u.startswith(XR))
print(f"Z-676: train {len(zt)} / val {len(zv)} / test {z_te.dataset.num_users};  "
      f"C2-hi: train {len(ht)} / val {len(hv)} / test {h_te.dataset.num_users}")
print(f"BOXRR train: Z {len(z_b)}, C2-hi {len(h_b)}, strict subset {h_b < z_b}; alyx train identical {({u for u in zt if u.startswith(ALYX)} == {u for u in ht if u.startswith(ALYX)})}; "
      f"validation identical on BOXRR+alyx {(zv == {u for u in hv if not u.startswith(XR)})}; C2-hi Across-XR train ids {h_x[0]}..{h_x[-1]} ({len(h_x)})")
assert len(zt) == len(ht), (len(zt), len(ht))
assert zv == hv, "the two arms must validate on identical people"
assert not any(u.startswith(XR) for u in hv) and not ({f"{XR}/{u}" for u in range(23, 32)} & (ht | hv))
assert h_b < z_b and len(z_b) - len(h_b) == SWAP
assert h_x == list(range(0, 23)) and z_te.dataset.num_users == h_te.dataset.num_users == 17
idx = h_tr.dataset.sample_index
counts = dict(zip(idx.dataset_names, torch.bincount(idx.window_dataset_ids, minlength=len(idx.dataset_names)).tolist()))
total = sum(counts.values())
dose = 100 * counts.get("CrossApplicationXR_Dataset", 0) / total
lists["c2hi_training_windows"] = counts
lists["c2hi_across_xr_dose_percent"] = round(dose, 2)
out.write_text(json.dumps(lists, indent=1), encoding="utf-8")
print(f"C2-hi training windows {counts}, total {total}, Across-XR dose {dose:.1f}%; {time.time() - t0:.0f}s")

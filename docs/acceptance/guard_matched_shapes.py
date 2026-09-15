"""The shape the guard fix exists for: ONE corpus, THREE disjoint user lists, through the
`validation_users` path that is new in the same identity step (73ecbf9232).

  C1  Across-XR alone: train 0-22, validate 23-31 (explicit), test 32-48 (exclude + test_on_excluded)
  C2  BOXRR + alyx + Across-XR: the same three lists on Across-XR, the 25% draw on the rest

Loader builds only, no training. The guard must return 0 with NON-EMPTY sets on all three
lists, and the lists must be exactly the registered ones - asserted on the directories the
loaders actually hold, not on the config.
"""
import contextlib, io, os, sys, time
ROOT = sys.argv[1]
MAIN = "/run/media/feng/Data/CalebProject/XRSec"
sys.path.insert(0, os.path.join(ROOT, "model"))
os.environ["XRSEC_SAMPLE_CACHE_DIR"] = MAIN + "/.cache/samples"
import torch
from dataset import create_dataloader_from_path, assert_evaluation_users_are_unseen, _user_dirs_of

ALYX = MAIN + "/processed_datasets/who_is_alyx/users"
BOXRR = MAIN + "/processed_datasets/BOXRR-23_Dataset/users"
XR = MAIN + "/processed_datasets/CrossApplicationXR_Dataset/users"
dev = torch.device("cpu")
common = dict(sample_time=10, sample_rate=20, samples_per_user=64, channels="full",
              window_stride=5, encoding="dyn", normalize="per_dataset",
              within_dataset_negatives=True, cross_session_positives=True, val_user_fraction=0.25)
excl = [f"{XR}/{u}" for u in range(32, 49)]
val = [f"{XR}/{u}" for u in range(23, 32)]


def xr_ids(dataset):
    return sorted(int(os.path.basename(d)) for d in _user_dirs_of(dataset) if d.startswith(XR))


for arm, dirs, expect in (("C1", [XR], (23, 9, 17)), ("C2", [BOXRR, ALYX, XR], (3072 + 23, 1024 + 9, 17))):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = create_dataloader_from_path(dirs, 256, dev, is_train=True, test_dir=[XR],
                                                 exclude_users=excl, swap_data=False, test_on_excluded=True,
                                                 seed=1, return_val=True, validation_users=val, **common)
    n = assert_evaluation_users_are_unseen(tr.dataset, te.dataset)
    counts = (tr.dataset.num_users, va.dataset.num_users, te.dataset.num_users)
    seen = (len(_user_dirs_of(tr.dataset)), len(_user_dirs_of(va.dataset)), len(_user_dirs_of(te.dataset)))
    print(f"{arm}: guard returned {n}; train/val/test users {counts}; dirs seen {seen}; "
          f"Across-XR ids train {xr_ids(tr.dataset)[:3]}..{xr_ids(tr.dataset)[-1]} ({len(xr_ids(tr.dataset))}), "
          f"val {xr_ids(va.dataset)}, test {xr_ids(te.dataset)[0]}..{xr_ids(te.dataset)[-1]} ({len(xr_ids(te.dataset))}); {time.time()-t0:.0f}s")
    assert n == 0 and counts == expect and all(s > 0 for s in seen) and seen == counts, (n, counts, seen)
    assert xr_ids(tr.dataset) == list(range(0, 23)) and xr_ids(va.dataset) == list(range(23, 32)) \
        and xr_ids(te.dataset) == list(range(32, 49))
    assert not (_user_dirs_of(tr.dataset) & _user_dirs_of(va.dataset)) and not (_user_dirs_of(va.dataset) & _user_dirs_of(te.dataset))
print("MATCHED SHAPES: guard returns 0 with non-empty sets; the three lists are exactly 0-22 / 23-31 / 32-48")

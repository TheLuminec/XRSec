"""The mirror check: the newly live unseen-users guard must return 0 on the two REAL shapes
this project builds, or enabling it blocks valid work.

  1. sweep.folds: same corpus both sides, exclude lists from build_folds, test_on_excluded
  2. cross-corpus: data_dirs BOXRR+alyx, test_dirs Across-XR with exclude 32-48 and
     test_on_excluded=true (the exact zero-shot instrument)

Two loader builds, no training. Prints the guard's return value and the user counts.
"""
import contextlib, io, os, sys, time
from types import SimpleNamespace
ROOT = sys.argv[1]
MAIN = "/run/media/feng/Data/CalebProject/XRSec"
sys.path.insert(0, os.path.join(ROOT, "model"))
os.environ["XRSEC_SAMPLE_CACHE_DIR"] = MAIN + "/.cache/samples"
import torch
from dataset import create_dataloader_from_path, assert_evaluation_users_are_unseen, _user_dirs_of
from sweep import build_folds

ALYX = MAIN + "/processed_datasets/who_is_alyx/users"
BOXRR = MAIN + "/processed_datasets/BOXRR-23_Dataset/users"
XR = MAIN + "/processed_datasets/CrossApplicationXR_Dataset/users"
dev = torch.device("cpu")
common = dict(sample_time=10, sample_rate=20, samples_per_user=64, channels="full",
              window_stride=5, encoding="dyn", normalize="per_dataset",
              within_dataset_negatives=True, cross_session_positives=True, val_user_fraction=0.25)

# --- shape 1: sweep.folds on one corpus (alyx, 76 users), fold 0 of 5
t0 = time.time()
cfg = SimpleNamespace(data_dirs=[ALYX], max_users=None)
folds = build_folds(cfg, 5, 67)
with contextlib.redirect_stdout(io.StringIO()):
    tr, va, te = create_dataloader_from_path([ALYX], 256, dev, is_train=True, test_dir=None,
                                             exclude_users=folds[0], swap_data=False,
                                             test_on_excluded=True, seed=67, return_val=True, **common)
n = assert_evaluation_users_are_unseen(tr.dataset, te.dataset)
print(f"shape 1 (sweep.folds, alyx fold 0/5): guard returned {n}; train users {tr.dataset.num_users}, "
      f"val {va.dataset.num_users}, test {te.dataset.num_users}; fold list {len(folds[0])}; "
      f"train dirs seen {len(_user_dirs_of(tr.dataset))}, test dirs seen {len(_user_dirs_of(te.dataset))}; {time.time()-t0:.0f}s")
assert n == 0 and te.dataset.num_users == len(folds[0]) and len(_user_dirs_of(tr.dataset)) == tr.dataset.num_users

# --- shape 2: cross-corpus, the zero-shot instrument
t0 = time.time()
excl = [f"{XR}/{u}" for u in range(32, 49)]
with contextlib.redirect_stdout(io.StringIO()):
    tr, va, te = create_dataloader_from_path([BOXRR, ALYX], 256, dev, is_train=True, test_dir=[XR],
                                             exclude_users=excl, swap_data=False,
                                             test_on_excluded=True, seed=1, return_val=True, **common)
n = assert_evaluation_users_are_unseen(tr.dataset, te.dataset)
print(f"shape 2 (BOXRR+alyx -> Across-XR 32-48): guard returned {n}; train users {tr.dataset.num_users}, "
      f"val {va.dataset.num_users}, test {te.dataset.num_users}; train dirs seen {len(_user_dirs_of(tr.dataset))}; {time.time()-t0:.0f}s")
assert n == 0 and te.dataset.num_users == 17 and tr.dataset.num_users + va.dataset.num_users == 4096
print("BOTH SHAPES: guard returns 0 on valid work, and sees the real user directories")

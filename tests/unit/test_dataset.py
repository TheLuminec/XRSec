import pathlib
import sys

import torch

import sampler
from dataset import SampleDataset, SiameseDataset, build_sample_index, generate_pair_manifest


FIXTURE_USERS_DIR = pathlib.Path(__file__).resolve().parents[1] / 'fixtures' / 'users'


def test_sample_dataset_returns_7x10_tensors(monkeypatch):
    monkeypatch.setattr(sampler.random, 'randint', lambda a, b: 0)

    dataset = SampleDataset(str(FIXTURE_USERS_DIR), sample_time=1, sample_rate=10)

    user_samples = dataset[0]
    assert user_samples.ndim == 3
    assert user_samples.shape[1:] == (7, 10)


def test_siamese_dataset_getitem_returns_pair_and_label(monkeypatch):
    monkeypatch.setattr(sampler.random, 'randint', lambda a, b: 0)

    dataset = SiameseDataset(str(FIXTURE_USERS_DIR), samples_per_user=3, sample_time=1, sample_rate=10)

    (x1, x2), y = dataset[0]

    assert x1.ndim == 2
    assert x2.ndim == 2
    assert y.ndim == 1
    assert x1.shape == (7, 10)
    assert x2.shape == (7, 10)


def test_generate_pair_manifest_is_deterministic(monkeypatch):
    monkeypatch.setattr(sampler.random, 'randint', lambda a, b: 0)

    sample_index = build_sample_index(str(FIXTURE_USERS_DIR), sample_time=1, sample_rate=10)
    manifest_a = generate_pair_manifest(sample_index, pairs_per_user=3, match_ratio=0.5, seed=17)
    manifest_b = generate_pair_manifest(sample_index, pairs_per_user=3, match_ratio=0.5, seed=17)

    for key in manifest_a:
        assert torch.equal(manifest_a[key], manifest_b[key])


def test_generate_pair_manifest_changes_with_seed(monkeypatch):
    monkeypatch.setattr(sampler.random, 'randint', lambda a, b: 0)

    sample_index = build_sample_index(str(FIXTURE_USERS_DIR), sample_time=1, sample_rate=10)
    manifest_a = generate_pair_manifest(sample_index, pairs_per_user=3, match_ratio=0.5, seed=17)
    manifest_b = generate_pair_manifest(sample_index, pairs_per_user=3, match_ratio=0.5, seed=18)

    assert not torch.equal(manifest_a["x1_indices"], manifest_b["x1_indices"]) or not torch.equal(
        manifest_a["x2_indices"], manifest_b["x2_indices"]
    )


def _multi_dataset_index(users_per_dataset=(2, 2)):
    """Sample index with users spread across two synthetic datasets."""
    import types
    import torch as T
    from dataset import SampleIndex

    fake = types.SimpleNamespace(
        sample_time=1, sample_rate=10, num_users=sum(users_per_dataset),
        dataset=[T.randn(5, 7, 10) for _ in range(sum(users_per_dataset))],
        dataset_names=[f"DS{i}" for i in range(len(users_per_dataset))],
        user_dataset_ids=[i for i, n in enumerate(users_per_dataset) for _ in range(n)],
    )
    return SampleIndex(fake)


def test_within_dataset_negatives_never_cross_datasets():
    """The shortcut fix: a negative partner must come from the same dataset."""
    index = _multi_dataset_index()
    user_of_window = {}
    for user, indices in enumerate(index.user_sample_indices):
        for i in indices.tolist():
            user_of_window[i] = user

    manifest = generate_pair_manifest(index, pairs_per_user=40, match_ratio=0.5, seed=3,
                                      within_dataset_negatives=True)
    negatives = manifest["labels"] == 0
    assert negatives.sum() > 0
    for a, b in zip(manifest["x1_indices"][negatives].tolist(), manifest["x2_indices"][negatives].tolist()):
        assert index.user_dataset_ids[user_of_window[a]] == index.user_dataset_ids[user_of_window[b]]


def test_unrestricted_negatives_do_cross_datasets():
    """Contrast: the default draws negatives from everywhere, which is the problem."""
    index = _multi_dataset_index()
    user_of_window = {i: u for u, ix in enumerate(index.user_sample_indices) for i in ix.tolist()}

    manifest = generate_pair_manifest(index, pairs_per_user=40, match_ratio=0.5, seed=3)
    negatives = manifest["labels"] == 0
    crossed = [
        index.user_dataset_ids[user_of_window[a]] != index.user_dataset_ids[user_of_window[b]]
        for a, b in zip(manifest["x1_indices"][negatives].tolist(), manifest["x2_indices"][negatives].tolist())
    ]
    assert any(crossed)


def test_within_dataset_negatives_is_a_noop_for_one_dataset():
    index = _multi_dataset_index(users_per_dataset=(4,))
    a = generate_pair_manifest(index, pairs_per_user=20, match_ratio=0.5, seed=5)
    b = generate_pair_manifest(index, pairs_per_user=20, match_ratio=0.5, seed=5,
                               within_dataset_negatives=True)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_lone_user_in_a_dataset_falls_back_to_positive_pairs():
    """A dataset with a single user has no within-dataset negative available."""
    index = _multi_dataset_index(users_per_dataset=(1, 3))
    manifest = generate_pair_manifest(index, pairs_per_user=10, match_ratio=0.5, seed=7,
                                      within_dataset_negatives=True)
    assert manifest["labels"].numel() > 0
    lone = manifest["anchor_user_ids"] == 0
    assert manifest["labels"][lone].min() == 1.0


def _sessioned_index(sessions_per_user=(3, 3), windows_per_session=4):
    """Sample index whose windows carry session provenance."""
    import types
    import torch as T
    from dataset import SampleIndex

    per_user, session_ids = [], []
    for count in sessions_per_user:
        total = count * windows_per_session
        per_user.append(T.randn(total, 7, 10))
        session_ids.append(T.arange(total) // windows_per_session)

    fake = types.SimpleNamespace(
        sample_time=1, sample_rate=10, num_users=len(sessions_per_user),
        dataset=per_user, dataset_names=["DS"],
        user_dataset_ids=[0] * len(sessions_per_user),
        session_ids=session_ids,
    )
    return SampleIndex(fake)


def _session_of(index):
    return {i: int(index.window_session_ids[i]) for i in range(index.sample_count)}


def test_sample_index_carries_session_provenance():
    index = _sessioned_index()
    assert index.window_session_ids.shape[0] == index.sample_count
    assert set(index.window_session_ids.tolist()) == {0, 1, 2}


def test_cross_session_positives_never_pair_within_one_session():
    """The validity fix: a positive must span two different recordings."""
    index = _sessioned_index()
    session = _session_of(index)

    manifest = generate_pair_manifest(index, pairs_per_user=40, match_ratio=1.0, seed=5,
                                      cross_session_positives=True)
    positives = manifest["labels"] == 1
    assert positives.sum() > 0
    for a, b in zip(manifest["x1_indices"][positives].tolist(),
                    manifest["x2_indices"][positives].tolist()):
        assert session[a] != session[b]


def test_default_positives_do_include_same_session_pairs():
    """Contrast: the default samples freely, which is the thing under suspicion."""
    index = _sessioned_index()
    session = _session_of(index)

    manifest = generate_pair_manifest(index, pairs_per_user=60, match_ratio=1.0, seed=5)
    positives = manifest["labels"] == 1
    same = [session[a] == session[b]
            for a, b in zip(manifest["x1_indices"][positives].tolist(),
                            manifest["x2_indices"][positives].tolist())]
    assert any(same)


def test_single_session_user_falls_back_instead_of_producing_nothing():
    """NJIT has one session per user; it must still contribute positives."""
    index = _sessioned_index(sessions_per_user=(1, 3))
    manifest = generate_pair_manifest(index, pairs_per_user=20, match_ratio=1.0, seed=7,
                                      cross_session_positives=True)
    lone = manifest["anchor_user_ids"] == 0
    assert lone.sum() > 0
    assert manifest["labels"][lone].min() == 1.0


def test_cross_session_positives_stay_within_the_same_user():
    index = _sessioned_index()
    user_of = {i: u for u, ix in enumerate(index.user_sample_indices) for i in ix.tolist()}
    manifest = generate_pair_manifest(index, pairs_per_user=30, match_ratio=1.0, seed=9,
                                      cross_session_positives=True)
    positives = manifest["labels"] == 1
    for a, b in zip(manifest["x1_indices"][positives].tolist(),
                    manifest["x2_indices"][positives].tolist()):
        assert user_of[a] == user_of[b]


def test_cross_session_positives_are_deterministic():
    index = _sessioned_index()
    a = generate_pair_manifest(index, pairs_per_user=20, match_ratio=0.5, seed=3,
                               cross_session_positives=True)
    b = generate_pair_manifest(index, pairs_per_user=20, match_ratio=0.5, seed=3,
                               cross_session_positives=True)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_center_position_removes_the_window_mean_from_position_only():
    """Centring must zero the position mean and leave orientation untouched."""
    import types
    from dataset import SampleIndex

    samples = torch.randn(6, 7, 10) + 5.0
    fake = types.SimpleNamespace(
        sample_time=1, sample_rate=10, num_users=1, dataset=[samples.clone()],
        dataset_names=["DS"], user_dataset_ids=[0], session_ids=[torch.zeros(6, dtype=torch.long)],
    )
    index = SampleIndex(fake, center_position=True)

    position = index.samples[:, 4:7, :]
    assert torch.allclose(position.mean(dim=2), torch.zeros(6, 3), atol=1e-5)
    # Quaternion channels are untouched, so their mean stays near the original +5.
    assert index.samples[:, :4, :].mean() > 4.0


def test_center_position_targets_the_right_channels_when_position_only():
    import types
    from dataset import SampleIndex, position_channel_slice

    assert position_channel_slice(7) == slice(4, 7)
    assert position_channel_slice(3) == slice(0, 3)

    samples = torch.randn(4, 3, 10) + 2.0
    fake = types.SimpleNamespace(
        sample_time=1, sample_rate=10, num_users=1, dataset=[samples.clone()],
        dataset_names=["DS"], user_dataset_ids=[0], session_ids=[torch.zeros(4, dtype=torch.long)],
        num_channels=3, channels="position",
    )
    index = SampleIndex(fake, center_position=True)
    assert torch.allclose(index.samples.mean(dim=2), torch.zeros(4, 3), atol=1e-5)


def test_center_position_is_off_by_default():
    import types
    from dataset import SampleIndex

    samples = torch.randn(4, 7, 10) + 5.0
    fake = types.SimpleNamespace(
        sample_time=1, sample_rate=10, num_users=1, dataset=[samples.clone()],
        dataset_names=["DS"], user_dataset_ids=[0], session_ids=[torch.zeros(4, dtype=torch.long)],
    )
    assert torch.allclose(SampleIndex(fake).samples, samples)


def _corpus(tmp_path, sizes):
    roots = []
    for name, count in sizes.items():
        root = tmp_path / name / "users"
        for user in range(count):
            (root / str(user)).mkdir(parents=True)
        roots.append(str(root))
    return roots


def test_user_subset_is_proportional_across_datasets(tmp_path):
    """
    Disambiguating identity count from data diversity requires the subsample to mirror
    the corpus, not over-represent whichever dataset sorts first.
    """
    from dataset import select_user_subset

    roots = _corpus(tmp_path, {"Big": 100, "Mid": 50, "Small": 10})
    kept = select_user_subset(roots, max_users=32, seed=7)

    assert len(kept) == 32
    counts = {name: sum(1 for u in kept if name in u) for name in ("Big", "Mid", "Small")}
    # 100/50/10 of 160, scaled to 32 -> 20 / 10 / 2
    assert counts == {"Big": 20, "Mid": 10, "Small": 2}


def test_user_subset_totals_exactly_the_requested_count(tmp_path):
    """Largest-remainder apportionment must not lose or gain users to rounding."""
    from dataset import select_user_subset

    roots = _corpus(tmp_path, {"A": 7, "B": 7, "C": 7})
    for target in (5, 11, 13, 20):
        assert len(select_user_subset(roots, max_users=target, seed=1)) == target


def test_user_subset_is_deterministic_and_seed_dependent(tmp_path):
    from dataset import select_user_subset

    roots = _corpus(tmp_path, {"A": 20, "B": 20})
    assert select_user_subset(roots, 10, seed=3) == select_user_subset(roots, 10, seed=3)
    assert select_user_subset(roots, 10, seed=3) != select_user_subset(roots, 10, seed=4)


def test_no_subsampling_when_not_needed(tmp_path):
    from dataset import select_user_subset

    roots = _corpus(tmp_path, {"A": 5})
    assert select_user_subset(roots, None, seed=1) is None
    assert select_user_subset(roots, 0, seed=1) is None
    assert select_user_subset(roots, 99, seed=1) is None, "asking for more than exist is a no-op"


def test_keep_users_filters_independently_of_swap_data(tmp_path):
    """
    A subsampled corpus must stay subsampled in the TEST split too, not just training,
    or the held-out group would quietly come from the full corpus.
    """
    import types
    from dataset import SampleDataset

    root = tmp_path / "DS" / "users"
    import numpy as np
    import pandas as pd
    for user in range(4):
        d = root / str(user)
        d.mkdir(parents=True)
        rows = 60
        pd.DataFrame({
            "SessionTime": np.arange(rows) * 0.1,
            "UnitQuaternion.x": np.zeros(rows), "UnitQuaternion.y": np.zeros(rows),
            "UnitQuaternion.z": np.zeros(rows), "UnitQuaternion.w": np.ones(rows),
            "HmdPosition.x": np.random.randn(rows), "HmdPosition.y": np.random.randn(rows),
            "HmdPosition.z": np.random.randn(rows),
        }).to_csv(d / "s.csv", index=False)

    keep = [str(root / "0"), str(root / "1"), str(root / "2")]
    excluded = [str(root / "0")]

    train = SampleDataset(str(root), sample_time=1, sample_rate=10,
                          exclude_users=excluded, swap_data=False, keep_users=keep)
    test = SampleDataset(str(root), sample_time=1, sample_rate=10,
                         exclude_users=excluded, swap_data=True, keep_users=keep)

    assert train.num_users == 2, "kept minus excluded"
    assert test.num_users == 1, "the excluded user, still inside the subsample"


def test_counts_users_that_cannot_form_cross_session_positives():
    """
    A user with one session falls back to same-session pairs, so a cross-session
    result is only as cross-session as the corpus allows. That has to be countable.
    """
    from dataset import count_single_session_users

    index = _sessioned_index(sessions_per_user=(3, 1, 2))
    assert count_single_session_users(index) == 1

    assert count_single_session_users(_sessioned_index(sessions_per_user=(4, 4))) == 0
    assert count_single_session_users(_sessioned_index(sessions_per_user=(1, 1, 1))) == 3


def test_single_session_count_is_zero_without_provenance():
    import types
    from dataset import count_single_session_users

    assert count_single_session_users(types.SimpleNamespace(window_session_ids=None)) == 0

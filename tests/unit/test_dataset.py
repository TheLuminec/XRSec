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

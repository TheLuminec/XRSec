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

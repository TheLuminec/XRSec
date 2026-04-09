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

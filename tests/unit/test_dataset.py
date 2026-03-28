import pathlib
import sys

from model import sampler
from model.dataset import SampleDataset, SiameseDataset


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

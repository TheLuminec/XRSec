import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'model'))
from sampler import Sampler


def _build_data(rows: int = 30) -> np.ndarray:
    data = np.zeros((rows, 8), dtype=np.float32)
    data[:, 0] = np.arange(rows, dtype=np.float32) * 0.1
    data[:, 1:] = np.arange(rows * 7, dtype=np.float32).reshape(rows, 7)
    return data


def test_get_data_point_closest_to_time_returns_nearest_index():
    data = _build_data(6)
    sampler = Sampler(data, sample_time=1, sample_rate=2, variance=0.0, index_randomness=0)

    idx = sampler._get_data_point_closest_to_time(0.26)

    assert idx == 3


def test_get_sample_raises_index_error_for_invalid_indices():
    data = _build_data(30)
    sampler = Sampler(data, sample_time=1, sample_rate=10, index_randomness=0)

    with pytest.raises(IndexError):
        sampler.get_sample(-1)

    with pytest.raises(IndexError):
        sampler.get_sample(sampler.sample_count)


def test_get_sample_has_expected_shape():
    sample_time = 2
    sample_rate = 5
    data = _build_data(40)
    sampler = Sampler(data, sample_time=sample_time, sample_rate=sample_rate, index_randomness=0)

    sample = sampler.get_sample(0)

    assert sample.shape == (sample_time * sample_rate, 8)

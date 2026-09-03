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


# --- bin-averaged resampling --------------------------------------------------

def _timed(rows, hz, channels=7):
    """Raw data at a known rate: time column plus `channels` channels."""
    data = np.zeros((rows, channels + 1), dtype=float)
    data[:, 0] = np.arange(rows) / hz
    if channels >= 7:
        data[:, 4] = 1.0                                  # w = 1, a valid quaternion
        data[:, 5:8] = np.arange(rows)[:, None] * 0.01    # position ramps
    else:
        data[:, 1:] = np.arange(rows)[:, None] * 0.01
    return data


def test_bin_resampling_removes_the_duplicate_frames_nearest_produces():
    """
    Below the native rate, nearest-point selection returns the same row repeatedly, so
    derived velocity is zero for those steps. Measured at 50.5% for ViewGauss at 20Hz.
    """
    data = _timed(rows=60, hz=10.0)            # asking 20Hz from 10Hz data

    nearest = Sampler(data, sample_time=2, sample_rate=20, resample="nearest").get_all_samples()
    binned = Sampler(data, sample_time=2, sample_rate=20, resample="bin").get_all_samples()

    duplicates = lambda w: (np.diff(w[:, :, 1:], axis=1) == 0).all(axis=2).mean()
    assert duplicates(nearest) > 0.3
    assert duplicates(binned) == 0.0


def test_bin_resampling_returns_the_interval_mean_not_a_member_of_it():
    """
    The value at each step should be the average of the rows inside that interval. A
    linear ramp cannot show this - averaging and picking give the same slope - so the
    signal here varies *within* each interval and the mean is a value no single row has.
    """
    rows, hz = 400, 100.0
    data = _timed(rows=rows, hz=hz)
    data[:, 5] = np.arange(rows) % 10          # 0..9 inside every 0.1s interval

    binned = Sampler(data, sample_time=1, sample_rate=10, resample="bin").get_all_samples()

    interior = binned[0, 2:-1, 5]
    assert np.allclose(interior, 4.5, atol=0.6), "not the mean of the interval"


def test_bin_resampling_attenuates_what_nearest_aliases():
    """
    Above the native rate, nearest keeps one row in ten and folds the discarded
    high-frequency content into what survives. Averaging is the anti-aliasing filter
    that belongs in front of a decimation, so the alternating component should
    largely disappear from the binned output and survive in the nearest one.
    """
    rows, hz = 400, 100.0
    data = _timed(rows=rows, hz=hz)
    alternating = np.where(np.arange(rows) % 2 == 0, 1.0, -1.0)
    data[:, 5] = alternating                   # pure Nyquist-rate content, zero mean

    kwargs = dict(sample_time=1, sample_rate=10)
    nearest = Sampler(data.copy(), **kwargs, resample="nearest").get_all_samples()[0, :, 5]
    binned = Sampler(data.copy(), **kwargs, resample="bin").get_all_samples()[0, :, 5]

    assert np.abs(binned).max() < 0.25, "high-frequency content survived averaging"
    assert np.abs(nearest).max() > 0.9, "nearest should alias it through untouched"


def test_bin_resampling_preserves_shape_and_time_column():
    data = _timed(rows=200, hz=50.0)
    nearest = Sampler(data, sample_time=2, sample_rate=20, resample="nearest")
    binned = Sampler(data, sample_time=2, sample_rate=20, resample="bin")

    assert binned.get_all_samples().shape == nearest.get_all_samples().shape
    times = binned.get_all_samples()[0, :, 0]
    assert np.all(np.diff(times) > 0), "the time column must stay increasing"


def test_bin_resampling_keeps_quaternions_unit_norm():
    rows = 300
    data = np.zeros((rows, 8))
    data[:, 0] = np.arange(rows) / 60.0
    angle = np.linspace(0, 1.5, rows)
    data[:, 3] = np.sin(angle / 2)      # z
    data[:, 4] = np.cos(angle / 2)      # w
    binned = Sampler(data, sample_time=2, sample_rate=20, resample="bin").get_all_samples()

    norms = np.linalg.norm(binned[:, :, 1:5], axis=2)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_bin_resampling_handles_the_quaternion_double_cover():
    """
    q and -q are the same rotation; averaging across a sign flip cancels instead of
    smoothing, which would collapse the quaternion toward zero before renormalising
    amplifies whatever noise is left.
    """
    rows = 240
    data = np.zeros((rows, 8))
    data[:, 0] = np.arange(rows) / 60.0
    data[:, 4] = 1.0
    data[::2] *= -1.0                     # flip sign on alternate rows
    data[:, 0] = np.arange(rows) / 60.0   # restore the time column after the flip

    binned = Sampler(data, sample_time=2, sample_rate=20, resample="bin").get_all_samples()
    norms = np.linalg.norm(binned[:, :, 1:5], axis=2)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # All rows are the identity rotation, so |w| must stay at 1 rather than average to 0.
    assert np.allclose(np.abs(binned[:, :, 4]), 1.0, atol=1e-5)


def test_position_only_data_is_resampled_too():
    data = _timed(rows=100, hz=50.0, channels=3)
    binned = Sampler(data, sample_time=1, sample_rate=10, resample="bin").get_all_samples()
    assert binned.shape[2] == 4          # time + 3 channels
    assert np.isfinite(binned).all()


def test_invalid_resample_mode_is_rejected():
    with pytest.raises(ValueError, match="resample must be"):
        Sampler(_timed(50, 20.0), sample_time=1, sample_rate=10, resample="spline")

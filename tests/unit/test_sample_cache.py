import os
import pathlib
import shutil

import pytest
import torch

import sample_cache
from dataset import build_sample_index


pytestmark = pytest.mark.unit


@pytest.fixture
def users_dir(tmp_path):
    """A private copy of the CSV fixtures, so mtime edits don't touch the originals."""
    source = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "users"
    destination = tmp_path / "dataset" / "users"
    shutil.copytree(source, destination)
    return destination


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XRSEC_SAMPLE_CACHE", "1")
    monkeypatch.setenv("XRSEC_SAMPLE_CACHE_DIR", str(tmp_path / "cache"))


def _build(users_dir):
    return build_sample_index(str(users_dir), sample_time=1, sample_rate=10)


def test_cached_index_matches_uncached_index(users_dir, monkeypatch):
    """The cache must be transparent: identical samples and identical user ranges."""
    monkeypatch.setenv("XRSEC_SAMPLE_CACHE", "0")
    uncached = _build(users_dir)

    monkeypatch.setenv("XRSEC_SAMPLE_CACHE", "1")
    cold = _build(users_dir)
    warm = _build(users_dir)

    assert torch.equal(uncached.samples, warm.samples)
    assert torch.equal(cold.samples, warm.samples)
    assert uncached.sample_count == warm.sample_count
    assert uncached.num_users == warm.num_users
    for expected, actual in zip(uncached.user_sample_indices, warm.user_sample_indices):
        assert torch.equal(expected, actual)


def test_second_build_hits_the_cache(users_dir):
    from dataset import SampleDataset

    cold = SampleDataset(str(users_dir), sample_time=1, sample_rate=10)
    warm = SampleDataset(str(users_dir), sample_time=1, sample_rate=10)

    assert cold.cache_misses == cold.num_users and cold.cache_hits == 0
    assert warm.cache_hits == warm.num_users and warm.cache_misses == 0


def test_editing_a_csv_invalidates_only_that_user(users_dir):
    from dataset import SampleDataset

    SampleDataset(str(users_dir), sample_time=1, sample_rate=10)

    edited = sorted(users_dir.iterdir())[0]
    csv_file = sorted(edited.glob("*.csv"))[0]
    stat = csv_file.stat()
    os.utime(csv_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    rebuilt = SampleDataset(str(users_dir), sample_time=1, sample_rate=10)
    assert rebuilt.cache_misses == 1
    assert rebuilt.cache_hits == rebuilt.num_users - 1


def test_sampling_resolution_is_part_of_the_cache_key(users_dir):
    a = build_sample_index(str(users_dir), sample_time=1, sample_rate=10)
    b = build_sample_index(str(users_dir), sample_time=1, sample_rate=5)

    assert a.samples.shape[2] == 10
    assert b.samples.shape[2] == 5


def test_corrupt_cache_entry_is_rebuilt(users_dir):
    from dataset import SampleDataset

    SampleDataset(str(users_dir), sample_time=1, sample_rate=10)
    for entry in sample_cache.cache_dir().glob("*.pt"):
        entry.write_bytes(b"not a torch file")

    rebuilt = SampleDataset(str(users_dir), sample_time=1, sample_rate=10)
    assert rebuilt.cache_misses == rebuilt.num_users
    assert rebuilt.sample_count > 0

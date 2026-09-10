"""The model/*.py additions batched on 2026-09-10 as one identity step."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset import dataset_tier, select_validation_users
import results_log

pytestmark = pytest.mark.unit


def test_the_converted_across_xr_directory_has_a_tier():
    assert dataset_tier("CrossApplicationXR_Dataset") == 1


def _corpus(tmp_path, name, users):
    root = tmp_path / name / "users"
    for u in users:
        (root / u).mkdir(parents=True)
    return str(root)


def test_explicit_validation_users_replace_the_draw_on_their_corpus_only(tmp_path):
    xr = _corpus(tmp_path, "CrossApplicationXR_Dataset", [str(i) for i in range(49)])
    other = _corpus(tmp_path, "Other", [f"u{i}" for i in range(20)])
    explicit = [str(Path(xr) / str(i)) for i in range(23, 32)]
    test_users = [str(Path(xr) / str(i)) for i in range(32, 49)]
    chosen = select_validation_users([xr, other], test_users, 0.25, seed=1, explicit=explicit)
    xr_chosen = sorted(int(Path(p).name) for p in chosen if p.startswith(xr))
    assert xr_chosen == list(range(23, 32))                 # exactly the published split
    other_chosen = [p for p in chosen if p.startswith(other)]
    assert len(other_chosen) == 5                            # the 25% draw still applies elsewhere
    assert not set(chosen) & set(test_users)


def test_explicit_validation_users_outside_the_corpus_or_excluded_are_ignored(tmp_path):
    xr = _corpus(tmp_path, "CrossApplicationXR_Dataset", [str(i) for i in range(10)])
    stray = str(tmp_path / "nowhere" / "users" / "7")
    excluded = str(Path(xr) / "3")
    chosen = select_validation_users(xr, [excluded], 0.0, seed=1,
                                     explicit=[stray, excluded, str(Path(xr) / "4")])
    assert chosen == [str(Path(xr) / "4")]


def test_no_explicit_users_and_zero_fraction_is_the_historical_behaviour(tmp_path):
    xr = _corpus(tmp_path, "A", [str(i) for i in range(10)])
    assert select_validation_users(xr, [], 0.0, seed=1) == []
    assert select_validation_users(xr, [], 0.0, seed=1, explicit=[]) == []


def test_the_fractional_draw_is_unchanged_without_explicit_users(tmp_path):
    xr = _corpus(tmp_path, "A", [str(i) for i in range(20)])
    before = select_validation_users(xr, [], 0.25, seed=3)
    after = select_validation_users(xr, [], 0.25, seed=3, explicit=None)
    assert before == after and len(before) == 5


def _cfg(**overrides):
    cfg = SimpleNamespace(
        mode="train", experiment_name="xrsec", seed=7,
        sample_time=2, sample_rate=10, embedding_dim=32, samples_per_user=64,
        batch_size=512, lr=0.001, data_dirs=["/x/DatasetA/users"], test_dirs=[],
        exclude_users=[], swap_data=False, test_on_excluded=True,
        save_path="/x/ckpt.pth", model_path="/x/ckpt.pth",
        extractor="bilstm", extractor_params=None,
        boosting=SimpleNamespace(enabled=False),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_every_row_carries_the_environment_and_the_validation_count(tmp_path):
    path = tmp_path / "shard.jsonl"
    results_log.append_run(_cfg(validation_users=["/x/DatasetA/users/9"]),
                           {"best_test_acc": 0.6, "best_epoch": 1}, "tag", results_path=path)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    env = results_log.environment()
    assert env["python_version"] and env["numpy_version"] and env["torch_version"]   # assert on the fixture
    for key in ("python_version", "numpy_version", "torch_version", "cuda_version",
                "device_name", "device_capability", "torch_arch_list"):
        assert row[key] == env[key], key
    assert row["num_validation_users"] == 1
    assert all(key in results_log.FIELDS for key in env)

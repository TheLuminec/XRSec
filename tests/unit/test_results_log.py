import csv
from types import SimpleNamespace

import pytest

import results_log


pytestmark = pytest.mark.unit


def _cfg(**overrides):
    cfg = SimpleNamespace(
        mode="train", experiment_name="xrsec", seed=7,
        sample_time=2, sample_rate=10, embedding_dim=32, samples_per_user=64,
        batch_size=512, lr=0.001, data_dirs=["/x/DatasetA/users"], test_dirs=[],
        exclude_users=[], swap_data=False, test_on_excluded=True,
        save_path="/x/ckpt.pth", model_path="/x/ckpt.pth",
        extractor="bilstm", extractor_params={"lstm_hidden": 128},
        boosting=SimpleNamespace(enabled=False),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _history():
    return {
        "train_loss": [0.7, 0.6], "train_acc": [0.5, 0.6],
        "test_loss": [0.7, 0.65], "test_acc": [0.52, 0.61],
        "best_test_acc": 0.61, "best_epoch": 2,
    }


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_appends_a_row_with_extractor_identity(tmp_path):
    path = tmp_path / "runs.csv"
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["extractor"] == "bilstm"
    assert rows[0]["extractor_params"] == "lstm_hidden=128"
    assert float(rows[0]["best_test_acc"]) == 0.61
    assert rows[0]["epochs_run"] == "2"


def test_appends_without_rewriting_when_schema_matches(tmp_path):
    path = tmp_path / "runs.csv"
    for _ in range(3):
        results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    assert len(_rows(path)) == 3


def test_boosted_and_test_results_are_summarized(tmp_path):
    path = tmp_path / "runs.csv"

    boosted = {
        "mode": "boosted",
        "best_checkpoint": "/x/best.pth",
        "round_summaries": [
            {"best_test_acc": 0.55, "best_epoch": 1},
            {"best_test_acc": 0.63, "best_epoch": 2},
        ],
        "round_histories": [_history(), _history()],
    }
    results_log.append_run(
        _cfg(boosting=SimpleNamespace(enabled=True, hard_fraction=0.6,
                                      candidate_pairs_per_user=1024, match_ratio=0.5)),
        boosted, dataset_tag="users", results_path=path,
    )
    results_log.append_run(_cfg(mode="test"), (0.69, 0.53), dataset_tag="users", results_path=path)

    rows = _rows(path)
    assert float(rows[0]["best_test_acc"]) == 0.63
    assert rows[0]["rounds_run"] == "2"
    assert rows[0]["hard_fraction"] == "0.6"
    assert rows[1]["mode"] == "test"
    assert float(rows[1]["final_test_acc"]) == 0.53


def test_schema_change_migrates_existing_rows(tmp_path, monkeypatch):
    """Adding a column must not corrupt or discard rows written earlier."""
    path = tmp_path / "runs.csv"
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    monkeypatch.setattr(results_log, "FIELDS", results_log.FIELDS + ["new_metric"])
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    rows = _rows(path)
    assert len(rows) == 2
    assert "new_metric" in rows[0]
    assert rows[0]["new_metric"] == ""          # backfilled blank
    assert rows[0]["extractor"] == "bilstm"     # original data intact


def test_columns_removed_from_schema_are_retained(tmp_path, monkeypatch):
    path = tmp_path / "runs.csv"
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    monkeypatch.setattr(results_log, "FIELDS", [f for f in results_log.FIELDS if f != "lr"])
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    rows = _rows(path)
    assert len(rows) == 2
    assert "lr" in rows[0], "dropping a column from FIELDS must not delete existing data"


def test_logging_failure_does_not_raise(tmp_path):
    """A finished run must never be lost to a logging error."""
    broken = SimpleNamespace(mode="train")  # missing nearly every attribute
    assert results_log.append_run(broken, _history(), "users", results_path=tmp_path / "r.csv") is None


def test_stray_header_from_a_union_merge_is_not_treated_as_data(tmp_path, monkeypatch):
    """
    results/runs.csv is union-merged across machines. A merge of two differing
    schemas can leave a duplicated header line mid-file; it must not become a row.
    """
    path = tmp_path / "runs.csv"
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    with path.open("a", newline="", encoding="utf-8") as handle:
        handle.write(",".join(results_log.FIELDS) + "\n")

    monkeypatch.setattr(results_log, "FIELDS", results_log.FIELDS + ["another_metric"])
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    rows = _rows(path)
    assert len(rows) == 2, "the duplicated header must be dropped, not migrated as data"
    assert all(row["mode"] == "train" for row in rows)

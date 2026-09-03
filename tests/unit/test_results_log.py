import csv
from types import SimpleNamespace

import pytest

import json
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


def test_curve_rows_are_recorded_one_per_k(tmp_path):
    """
    A k-curve that only ever existed in stdout would repeat the mistake this file was
    written to fix.
    """
    path = tmp_path / "runs.csv"
    curve = [
        {"k": 1, "pairs": 640, "positive_fraction": 0.5, "auc": 0.71,
         "eer": 0.34, "accuracy_at_eer": 0.66},
        {"k": 8, "pairs": 640, "positive_fraction": 0.5, "auc": 0.78,
         "eer": 0.28, "accuracy_at_eer": 0.72},
    ]
    for row in curve:
        results_log.append_run(_cfg(mode="curve"), row, dataset_tag="users", results_path=path)

    rows = _rows(path)
    assert [r["template_k"] for r in rows] == ["1", "8"]
    assert float(rows[1]["selected_test_auc"]) == 0.78
    assert float(rows[1]["selected_test_eer"]) == 0.28
    assert float(rows[1]["eval_positive_fraction"]) == 0.5


# --- per-machine JSONL shards -------------------------------------------------
#
# The committed log is union-merged across three machines. These cover the property
# that makes that safe, not just that writing works.

def _shard(tmp_path):
    return tmp_path / "runs" / "desktop-c.jsonl"


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_shard_append_writes_one_self_describing_line_per_run(tmp_path):
    path = _shard(tmp_path)
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    rows = _lines(path)
    assert len(rows) == 2
    assert all(row["seed"] == 7 and row["sample_time"] == 2 for row in rows)


def test_every_run_gets_a_distinct_id(tmp_path):
    """Union merge coalesces identical lines, so two runs must never produce one."""
    path = _shard(tmp_path)
    for _ in range(3):
        results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)
    ids = [row["run_id"] for row in _lines(path)]
    assert len(set(ids)) == 3


def test_adding_a_column_never_rewrites_an_existing_line(tmp_path, monkeypatch):
    """
    The CSV writer migrates the file in place, which is what put two layouts in front
    of the union driver. A shard must only ever grow at the end.
    """
    path = _shard(tmp_path)
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)
    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(results_log, "FIELDS", results_log.FIELDS + ["new_metric"])
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    assert path.read_text(encoding="utf-8").startswith(before), "an existing line changed"


def test_union_merging_two_schemas_keeps_every_field_on_the_right_row(tmp_path, monkeypatch):
    """
    Regression for the corruption: HEAD carried 57 columns and the other machine 56,
    so a line-based union filed every row from one side under the other's header -
    537 rows, 237 duplicated, seed 67 reading as 2. With self-describing lines the
    same union is correct, because concatenation is all a union merge does.
    """
    monkeypatch.setattr(results_log, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(results_log, "LEGACY_RESULTS_PATH", tmp_path / "absent.csv")

    narrow = tmp_path / "runs" / "narrow.jsonl"
    results_log.append_run(_cfg(seed=67), _history(), dataset_tag="users", results_path=narrow)

    monkeypatch.setattr(results_log, "FIELDS", results_log.FIELDS + ["late_addition"])
    wide = tmp_path / "runs" / "wide.jsonl"
    results_log.append_run(_cfg(seed=2), _history(), dataset_tag="users", results_path=wide)

    # Exactly what `merge=union` does to two versions of a file.
    merged = tmp_path / "runs" / "merged.jsonl"
    merged.write_text(
        narrow.read_text(encoding="utf-8") + wide.read_text(encoding="utf-8"), encoding="utf-8")
    narrow.unlink()
    wide.unlink()

    rows = results_log.load_runs()
    assert len(rows) == 2
    assert sorted(row["seed"] for row in rows) == [2, 67]
    assert all(row["sample_time"] == 2 for row in rows), "fields shifted across the merge"


def test_load_runs_reads_the_frozen_csv_and_the_shards_together(tmp_path):
    legacy = tmp_path / "runs.csv"
    results_log.append_run(_cfg(), _history(), dataset_tag="legacy", results_path=legacy)

    shard = _shard(tmp_path)
    results_log.append_run(_cfg(), _history(), dataset_tag="shard", results_path=shard)

    import unittest.mock as mock
    with mock.patch.object(results_log, "LEGACY_RESULTS_PATH", legacy),          mock.patch.object(results_log, "RUNS_DIR", tmp_path / "runs"):
        tags = [row["dataset_tag"] for row in results_log.load_runs()]
    assert sorted(tags) == ["legacy", "shard"]


def test_combined_csv_has_one_header_covering_every_column(tmp_path, monkeypatch):
    monkeypatch.setattr(results_log, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(results_log, "LEGACY_RESULTS_PATH", tmp_path / "absent.csv")
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=_shard(tmp_path))

    out = results_log.write_combined_csv(tmp_path / "runs_all.csv")
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["seed"] == "7"
    assert rows[0]["run_id"]


def test_a_damaged_line_does_not_hide_the_rest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(results_log, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(results_log, "LEGACY_RESULTS_PATH", tmp_path / "absent.csv")
    path = _shard(tmp_path)
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    results_log.append_run(_cfg(), _history(), dataset_tag="users", results_path=path)

    assert len(results_log.load_runs()) == 2
    assert "unreadable line" in capsys.readouterr().out

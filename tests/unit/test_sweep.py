import csv
import json

import pytest
from omegaconf import OmegaConf

import sweep


pytestmark = pytest.mark.unit


def _cfg(tmp_path, **sweep_overrides):
    base = {
        "mode": "sweep", "experiment_name": "xrsec", "seed": 7, "epochs": 10,
        "lr": 0.001, "batch_size": 512, "embedding_dim": 32, "samples_per_user": 64,
        "sample_time": 2, "sample_rate": 10, "num_workers": 0,
        "data_dirs": ["/x/DatasetA/users"], "test_dirs": [], "exclude_users": [],
        "swap_data": False, "test_on_excluded": True,
        "extractor": "bilstm", "extractor_params": None,
        "save_path": "/x/ckpt.pth", "model_path": "/x/ckpt.pth",
        "graph": False, "graph_path": "/x/g.png",
        "boosting": {"enabled": False, "artifact_root": "boosting"},
        "sweep": {
            "extractors": "config", "grid": "auto", "strategy": "grid",
            "max_runs": None, "seed": None, "epochs": None, "dry_run": False,
            "artifact_root": str(tmp_path / "sweeps"), "resume": True,
        },
    }
    base["sweep"].update(sweep_overrides)
    return OmegaConf.create(base)


@pytest.fixture(autouse=True)
def no_results_logging(monkeypatch):
    """Keep tests out of the real results/runs.csv."""
    monkeypatch.setattr(sweep.results_log, "append_run", lambda *a, **k: None)


def _history(acc):
    return {"train_loss": [0.7], "train_acc": [0.5], "test_loss": [0.7],
            "test_acc": [acc], "best_test_acc": acc, "best_epoch": 1}


# --- configuration expansion -------------------------------------------------

def test_auto_grid_uses_the_extractor_declared_search_space(tmp_path):
    import feature_extractor as fe

    configurations = sweep.build_configurations(_cfg(tmp_path))
    expected = 1
    for values in fe.search_space("bilstm").values():
        expected *= len(values)

    assert len(configurations) == expected
    assert {c["extractor"] for c in configurations} == {"bilstm"}


def test_explicit_grid_expands_every_combination(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001], "extractor_params.lstm_hidden": [32, 64]})
    configurations = sweep.build_configurations(cfg)

    assert len(configurations) == 4
    assert {tuple(sorted(c["overrides"].items())) for c in configurations} == {
        (("extractor_params.lstm_hidden", 32), ("lr", 0.01)),
        (("extractor_params.lstm_hidden", 64), ("lr", 0.01)),
        (("extractor_params.lstm_hidden", 32), ("lr", 0.001)),
        (("extractor_params.lstm_hidden", 64), ("lr", 0.001)),
    }


def test_extractors_all_covers_the_registry(tmp_path):
    import feature_extractor as fe

    cfg = _cfg(tmp_path, extractors="all", grid={"lr": [0.01]})
    configurations = sweep.build_configurations(cfg)

    assert {c["extractor"] for c in configurations} == set(fe.available())


def test_max_runs_caps_the_grid(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001, 0.0001]}, max_runs=2)
    assert len(sweep.build_configurations(cfg)) == 2


def test_random_strategy_is_deterministic_and_samples_without_replacement(tmp_path):
    kwargs = dict(grid={"lr": [1, 2, 3, 4, 5, 6, 7, 8]}, strategy="random", max_runs=4, seed=99)
    first = sweep.build_configurations(_cfg(tmp_path, **kwargs))
    second = sweep.build_configurations(_cfg(tmp_path, **kwargs))

    assert [c["id"] for c in first] == [c["id"] for c in second]
    assert len({c["id"] for c in first}) == 4


def test_invalid_strategy_and_grid_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        sweep.build_configurations(_cfg(tmp_path, strategy="bogus"))
    with pytest.raises(ValueError, match="non-empty list"):
        sweep.build_configurations(_cfg(tmp_path, grid={"lr": 0.01}))


def test_configuration_ids_are_stable_and_distinct(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001]})
    a = sweep.build_configurations(cfg)
    b = sweep.build_configurations(cfg)

    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert len({c["id"] for c in a}) == 2


# --- per-run config ----------------------------------------------------------

def test_apply_configuration_routes_overrides_to_the_right_place(tmp_path):
    cfg = _cfg(tmp_path, epochs=3)
    configuration = {
        "id": "abc123", "extractor": "bilstm",
        "overrides": {"lr": 0.01, "extractor_params.lstm_hidden": 128},
    }

    run_cfg = sweep.apply_configuration(cfg, configuration, tmp_path / "root", "sweep1")

    assert run_cfg.mode == "train"                       # sweep runs are training runs
    assert run_cfg.lr == 0.01                            # top-level axis
    assert run_cfg.extractor_params == {"lstm_hidden": 128}   # namespaced axis
    assert run_cfg.epochs == 3                           # sweep.epochs override
    assert run_cfg.sweep_id == "sweep1"
    assert "abc123" in run_cfg.save_path
    assert cfg.lr == 0.001, "the base config must not be mutated"


def test_each_configuration_gets_its_own_artifact_paths(tmp_path):
    cfg = _cfg(tmp_path)
    paths = {
        sweep.apply_configuration(
            cfg, {"id": cid, "extractor": "bilstm", "overrides": {}}, tmp_path / "r", "s"
        ).save_path
        for cid in ("aaa", "bbb")
    }
    assert len(paths) == 2


# --- running -----------------------------------------------------------------

def test_runs_every_configuration_and_ranks_them(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001, 0.0001]})
    seen = []

    def fake_train(run_cfg):
        seen.append(run_cfg.lr)
        return _history({0.01: 0.7, 0.001: 0.6, 0.0001: 0.5}[run_cfg.lr])

    result = sweep.run_sweep(cfg, train_fn=fake_train)

    assert sorted(seen) == [0.0001, 0.001, 0.01]
    assert result["best"]["best_test_acc"] == 0.7
    assert result["best"]["overrides"]["lr"] == 0.01


def test_a_failing_configuration_does_not_abort_the_sweep(tmp_path):
    """Generated extractors will fail on some combinations; the rest must still run."""
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001, 0.0001]})

    def flaky_train(run_cfg):
        if run_cfg.lr == 0.001:
            raise RuntimeError("CUDA out of memory")
        return _history(0.6)

    result = sweep.run_sweep(cfg, train_fn=flaky_train)

    statuses = [r["status"] for r in result["records"]]
    assert statuses.count("ok") == 2
    assert statuses.count("failed") == 1
    failed = next(r for r in result["records"] if r["status"] == "failed")
    assert "CUDA out of memory" in failed["error"]
    assert result["best"]["best_test_acc"] == 0.6


def test_resume_skips_completed_configurations(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001]})
    calls = []

    def counting_train(run_cfg):
        calls.append(run_cfg.lr)
        return _history(0.6)

    sweep.run_sweep(cfg, train_fn=counting_train)
    assert len(calls) == 2

    sweep.run_sweep(cfg, train_fn=counting_train)
    assert len(calls) == 2, "a resumed sweep must not retrain completed configurations"


def test_resume_retries_previously_failed_configurations(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]})
    attempts = []

    def failing(run_cfg):
        attempts.append(1)
        raise RuntimeError("boom")

    sweep.run_sweep(cfg, train_fn=failing)
    sweep.run_sweep(cfg, train_fn=failing)
    assert len(attempts) == 2


def test_resume_false_reruns_everything(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]}, resume=False)
    calls = []

    def counting_train(run_cfg):
        calls.append(1)
        return _history(0.6)

    sweep.run_sweep(cfg, train_fn=counting_train)
    sweep.run_sweep(cfg, train_fn=counting_train)
    assert len(calls) == 2


def test_dry_run_trains_nothing_but_reports_the_plan(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001]}, dry_run=True)

    def must_not_run(run_cfg):
        raise AssertionError("dry_run must not train")

    result = sweep.run_sweep(cfg, train_fn=must_not_run)
    assert result["dry_run"] is True
    assert len(result["configurations"]) == 2


def test_writes_state_and_summary_files(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001]})
    result = sweep.run_sweep(cfg, train_fn=lambda run_cfg: _history(0.6))

    summary = list(csv.DictReader(open(result["summary_path"], newline="", encoding="utf-8")))
    assert len(summary) == 2
    assert summary[0]["extractor"] == "bilstm"
    assert summary[0]["status"] == "ok"

    state_files = list((tmp_path / "sweeps").rglob("sweep_state.json"))
    assert state_files, "sweep state must be persisted for resume"
    assert len(json.loads(state_files[0].read_text(encoding="utf-8"))["records"]) == 2


def test_boosted_sweep_runs_are_scored_by_best_round(tmp_path):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]})
    boosted = {
        "mode": "boosted",
        "round_summaries": [{"best_test_acc": 0.55}, {"best_test_acc": 0.66}],
    }
    result = sweep.run_sweep(cfg, train_fn=lambda run_cfg: boosted)
    assert result["best"]["best_test_acc"] == 0.66

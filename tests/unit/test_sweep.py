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


# --- cross-validation over user folds -----------------------------------------

@pytest.fixture
def user_tree(tmp_path):
    """A dataset directory with 6 users, enough to fold."""
    root = tmp_path / "DS" / "users"
    for user in range(6):
        (root / str(user)).mkdir(parents=True)
    return root


def test_folds_partition_every_user_exactly_once(tmp_path, user_tree):
    cfg = _cfg(tmp_path)
    cfg.data_dirs = [str(user_tree)]

    folds = sweep.build_folds(cfg, 3, seed=1)

    assert len(folds) == 3
    flat = [u for group in folds for u in group]
    assert len(flat) == 6 and len(set(flat)) == 6, "folds must be disjoint and cover everyone"


def test_folds_are_deterministic(tmp_path, user_tree):
    cfg = _cfg(tmp_path)
    cfg.data_dirs = [str(user_tree)]
    assert sweep.build_folds(cfg, 3, seed=7) == sweep.build_folds(cfg, 3, seed=7)
    assert sweep.build_folds(cfg, 3, seed=7) != sweep.build_folds(cfg, 3, seed=8)


def test_too_many_folds_is_rejected(tmp_path, user_tree):
    cfg = _cfg(tmp_path)
    cfg.data_dirs = [str(user_tree)]
    with pytest.raises(ValueError, match="at least that many users"):
        sweep.build_folds(cfg, 99, seed=1)


def test_fold_run_overrides_the_held_out_users(tmp_path, user_tree):
    """Each fold must evaluate on its own group, via the leave-users-out convention."""
    cfg = _cfg(tmp_path)
    cfg.data_dirs = [str(user_tree)]
    held = [str(user_tree / "0"), str(user_tree / "1")]

    run_cfg = sweep.apply_configuration(
        cfg, {"id": "abc", "extractor": "bilstm", "overrides": {}},
        tmp_path / "root", "s1", fold=2, held_out_users=held,
    )

    assert list(run_cfg.exclude_users) == held
    assert run_cfg.swap_data is False
    assert run_cfg.test_on_excluded is True
    assert run_cfg.fold == 2
    assert "fold2" in run_cfg.save_path


def test_cross_validated_sweep_reports_mean_and_spread(tmp_path, user_tree):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001]}, folds=3)
    cfg.data_dirs = [str(user_tree)]

    # Score depends on the fold, so the spread is non-zero and checkable.
    scores = {0: 0.60, 1: 0.70, 2: 0.80}

    def fake_train(run_cfg):
        return _history(scores[int(run_cfg.fold)] + (0.05 if run_cfg.lr == 0.01 else 0.0))

    result = sweep.run_sweep(cfg, train_fn=fake_train)

    assert len(result["records"]) == 2, "one aggregated row per configuration"
    for record in result["records"]:
        assert record["folds_completed"] == 3
        assert record["fold_std"] == pytest.approx(0.0816, abs=1e-3)
    best = result["best"]
    assert best["best_test_acc"] == pytest.approx(0.75, abs=1e-6)
    assert best["overrides"]["lr"] == 0.01


def test_cross_validated_sweep_runs_every_configuration_on_every_fold(tmp_path, user_tree):
    cfg = _cfg(tmp_path, grid={"lr": [0.01, 0.001]}, folds=3)
    cfg.data_dirs = [str(user_tree)]
    seen = []

    def fake_train(run_cfg):
        seen.append((run_cfg.lr, int(run_cfg.fold)))
        return _history(0.6)

    sweep.run_sweep(cfg, train_fn=fake_train)
    assert len(seen) == 6 and len(set(seen)) == 6


def test_a_failed_fold_does_not_discard_the_others(tmp_path, user_tree):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]}, folds=3)
    cfg.data_dirs = [str(user_tree)]

    def flaky(run_cfg):
        if int(run_cfg.fold) == 1:
            raise RuntimeError("boom")
        return _history(0.66)

    result = sweep.run_sweep(cfg, train_fn=flaky)
    record = result["records"][0]
    assert record["status"] == "ok"
    assert record["folds_completed"] == 2
    assert record["best_test_acc"] == pytest.approx(0.66)


def test_cross_validated_resume_skips_completed_folds(tmp_path, user_tree):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]}, folds=3)
    cfg.data_dirs = [str(user_tree)]
    calls = []

    def counting(run_cfg):
        calls.append(1)
        return _history(0.6)

    sweep.run_sweep(cfg, train_fn=counting)
    sweep.run_sweep(cfg, train_fn=counting)
    assert len(calls) == 3, "a resumed cross-validated sweep must not retrain finished folds"


def test_folds_are_stratified_across_datasets(tmp_path):
    """
    Pooled datasets are very uneven and differ in difficulty, so a fold heavy in one
    of them measures something different from its neighbours and inflates the fold
    spread - the statistic that decides whether a result is real.
    """
    roots = []
    for name, count in (("Big", 20), ("Small", 5)):
        root = tmp_path / name / "users"
        for user in range(count):
            (root / str(user)).mkdir(parents=True)
        roots.append(str(root))

    cfg = _cfg(tmp_path)
    cfg.data_dirs = roots
    folds = sweep.build_folds(cfg, 5, seed=3)

    for group in folds:
        big = sum(1 for u in group if "Big" in u)
        small = sum(1 for u in group if "Small" in u)
        assert big == 4, f"expected 20/5 = 4 Big users per fold, got {big}"
        assert small == 1, f"expected 5/5 = 1 Small user per fold, got {small}"


def test_stratification_handles_counts_that_do_not_divide_evenly(tmp_path):
    root = tmp_path / "DS" / "users"
    for user in range(7):
        (root / str(user)).mkdir(parents=True)
    cfg = _cfg(tmp_path)
    cfg.data_dirs = [str(root)]

    folds = sweep.build_folds(cfg, 3, seed=1)
    sizes = sorted(len(g) for g in folds)
    assert sizes == [2, 2, 3]
    flat = [u for g in folds for u in g]
    assert len(flat) == 7 and len(set(flat)) == 7


def test_fold_composition_is_reported(tmp_path):
    roots = []
    for name, count in (("Alpha", 6), ("Beta", 4)):
        root = tmp_path / name / "users"
        for user in range(count):
            (root / str(user)).mkdir(parents=True)
        roots.append(str(root))
    cfg = _cfg(tmp_path)
    cfg.data_dirs = roots

    text = sweep.describe_fold_composition(sweep.build_folds(cfg, 2, seed=5))
    assert "Alpha=3" in text and "Beta=2" in text
    assert "fold 0" in text and "fold 1" in text


# --- sweep identity: the collision that reported one sweep's numbers as another's ---

def _ids(tmp_path, **cfg_overrides):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]})
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    return sweep.config_identity(cfg, sweep.build_configurations(cfg))


def test_identity_is_stable_for_an_identical_sweep(tmp_path):
    assert _ids(tmp_path) == _ids(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("max_users", 48),
    ("objective", "identity_softmax"),
    ("normalize", "global"),
    ("sample_time", 5),
    ("sample_rate", 40),
    ("embedding_dim", 64),
    ("val_user_fraction", 0.25),
    ("cross_session_positives", True),
    ("center_position", True),
    ("channels", "position"),
    ("samples_per_user", 128),
    ("seed", 999),
])
def test_every_experimental_key_changes_the_identity(tmp_path, key, value):
    """
    The real bug: identity was extractor names plus grid overrides only, so a sweep
    differing in max_users reused the previous sweep's state, skipped all its runs as
    complete, and printed the previous sweep's numbers under the new banner.
    """
    assert _ids(tmp_path) != _ids(tmp_path, **{key: value}), (
        f"changing {key} must produce a different sweep identity, or the two sweeps "
        "share a state file and the second silently reports the first's results"
    )


def test_data_dirs_change_the_identity(tmp_path):
    assert _ids(tmp_path) != _ids(tmp_path, data_dirs=["/x/DatasetA/users", "/x/DatasetB/users"])


def test_operational_keys_do_not_change_the_identity(tmp_path):
    """Where artifacts land, or whether we resume, is not part of what is measured."""
    base = _cfg(tmp_path, grid={"lr": [0.01]})
    other = _cfg(tmp_path, grid={"lr": [0.01]}, artifact_root="/somewhere/else",
                 resume=False, dry_run=True)
    configurations = sweep.build_configurations(base)
    assert sweep.config_identity(base, configurations) == sweep.config_identity(other, configurations)


def test_identity_is_portable_across_machines(tmp_path):
    """Absolute paths differ per checkout; the same experiment must still match."""
    a = _ids(tmp_path, data_dirs=["/home/alice/repo/processed_datasets/DS/users"])
    b = _ids(tmp_path, data_dirs=["C:/Users/bob/GIT/repo/processed_datasets/DS/users"])
    assert a == b


def test_state_from_a_different_configuration_is_refused(tmp_path, capsys):
    """
    Belt and braces: a state file written by older code carries no identity, or a
    different one. Trusting the directory name is what caused the silent skip.
    """
    cfg = _cfg(tmp_path, grid={"lr": [0.01]})
    calls = []

    def counting(run_cfg):
        calls.append(1)
        return _history(0.6)

    result = sweep.run_sweep(cfg, train_fn=counting)
    assert len(calls) == 1

    state_file = next((tmp_path / "sweeps").rglob("sweep_state.json"))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["config_identity"] = "somethingelse"
    state_file.write_text(json.dumps(state), encoding="utf-8")

    sweep.run_sweep(cfg, train_fn=counting)
    assert len(calls) == 2, "a mismatched state file must not be resumed from"
    assert "Ignoring it and starting fresh" in capsys.readouterr().out


def test_state_without_an_identity_is_refused(tmp_path, capsys):
    cfg = _cfg(tmp_path, grid={"lr": [0.01]})
    calls = []
    sweep.run_sweep(cfg, train_fn=lambda c: (calls.append(1), _history(0.6))[1])

    state_file = next((tmp_path / "sweeps").rglob("sweep_state.json"))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    del state["config_identity"]
    state_file.write_text(json.dumps(state), encoding="utf-8")

    sweep.run_sweep(cfg, train_fn=lambda c: (calls.append(1), _history(0.6))[1])
    assert len(calls) == 2
    assert "cannot be verified" in capsys.readouterr().out

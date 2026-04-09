import json
import pathlib
from types import SimpleNamespace

from train import train


FIXTURE_USERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "users"


def _base_args(tmp_path, boosting_enabled):
    return SimpleNamespace(
        mode="train",
        seed=13,
        epochs=1,
        lr=0.001,
        batch_size=2,
        num_workers=0,
        embedding_dim=8,
        sample_time=1,
        sample_rate=10,
        samples_per_user=2,
        data_dirs=[str(FIXTURE_USERS_DIR)],
        test_dirs=[],
        exclude_users=[],
        swap_data=False,
        test_on_excluded=False,
        experiment_name="xrsec-test",
        save_path=str(tmp_path / ("best_overall.pth" if boosting_enabled else "standard_best.pth")),
        model_path=str(tmp_path / "unused.pth"),
        graph=False,
        graph_path=str(tmp_path / "unused.png"),
        boosting=SimpleNamespace(
            enabled=boosting_enabled,
            rounds=2,
            round_epochs=1,
            hard_fraction=0.5,
            refresh_fraction=0.5,
            candidate_pairs_per_user=4,
            match_ratio=0.5,
            artifact_root=str(tmp_path / "boosting"),
            resume="none",
        ),
    )


def test_boosted_training_creates_round_artifacts(tmp_path):
    args = _base_args(tmp_path, boosting_enabled=True)

    result = train(args)

    artifact_root = pathlib.Path(args.boosting.artifact_root)
    assert result["mode"] == "boosted"
    assert (artifact_root / "rounds" / "round_000_best.pth").exists()
    assert (artifact_root / "rounds" / "round_000_last.pth").exists()
    assert (artifact_root / "rounds" / "round_001_best.pth").exists()
    assert (artifact_root / "rounds" / "round_001_last.pth").exists()
    assert pathlib.Path(args.save_path).exists()
    assert not (artifact_root / "sample_index.pt").exists()
    assert not (artifact_root / "validation_pairs.pt").exists()
    assert not (artifact_root / "rounds" / "round_000_train_pairs.pt").exists()

    with (artifact_root / "boost_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    assert state["mode"] == "complete"
    assert len(state["round_summaries"]) == 2
    assert state["best_checkpoint"].endswith(".pth")


def test_boosted_training_is_reproducible_from_seed(tmp_path):
    run_a = _base_args(tmp_path / "run_a", boosting_enabled=True)
    run_b = _base_args(tmp_path / "run_b", boosting_enabled=True)

    result_a = train(run_a)
    result_b = train(run_b)

    assert [summary["best_epoch"] for summary in result_a["round_summaries"]] == [
        summary["best_epoch"] for summary in result_b["round_summaries"]
    ]
    assert [summary["best_test_acc"] for summary in result_a["round_summaries"]] == [
        summary["best_test_acc"] for summary in result_b["round_summaries"]
    ]
    assert [summary["train_pairs"] for summary in result_a["round_summaries"]] == [
        summary["train_pairs"] for summary in result_b["round_summaries"]
    ]
    assert [summary["manifest_meta"] for summary in result_a["round_summaries"]] == [
        summary["manifest_meta"] for summary in result_b["round_summaries"]
    ]
    assert pathlib.Path(result_a["best_checkpoint"]).name == pathlib.Path(result_b["best_checkpoint"]).name


def test_standard_training_path_still_saves_checkpoint(tmp_path):
    args = _base_args(tmp_path, boosting_enabled=False)

    history = train(args)

    assert pathlib.Path(args.save_path).exists()
    assert len(history["train_loss"]) == 1
    assert len(history["test_acc"]) == 1

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


def test_boosting_is_retired_and_refuses_to_run(tmp_path):
    """
    Boosting is retired rather than fixed: best-round selection reads the set it
    reports, its artifact root cannot resume, and it is pairwise-only so the
    identity_softmax objective (+6.5 points) does not apply. A boosted number is
    therefore not comparable to a current one, so it must refuse rather than produce
    one.
    """
    import pytest

    args = _base_args(tmp_path, boosting_enabled=True)

    with pytest.raises(RuntimeError, match="boosting is retired"):
        train(args)

    assert not pathlib.Path(args.save_path).exists(), "a retired path must not write artifacts"


def test_the_refusal_explains_what_to_use_instead(tmp_path):
    import pytest

    from train import BOOSTING_RETIRED

    with pytest.raises(RuntimeError) as raised:
        train(_base_args(tmp_path, boosting_enabled=True))

    message = str(raised.value)
    assert "identity_softmax" in message, "the refusal must name the supported path"
    assert "val_user_fraction" in message
    assert message == BOOSTING_RETIRED


def test_standard_training_path_still_saves_checkpoint(tmp_path):
    args = _base_args(tmp_path, boosting_enabled=False)

    history = train(args)

    assert pathlib.Path(args.save_path).exists()
    assert len(history["train_loss"]) == 1
    assert len(history["test_acc"]) == 1

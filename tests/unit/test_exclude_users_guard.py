"""
A separate evaluation corpus must never lose users to the training split's exclude list.

The config default names five VR_User_Behavior users in exclude_users. Every
cross-corpus run of the generalisation programme inherited it with test_on_excluded=false,
and scored VR_User_Behavior on 43 users while everyone read 48. The loader now refuses
that configuration rather than warning.
"""
import pathlib
import shutil

import pytest
import torch

from dataset import create_dataloader_from_path, refuse_excluded_users_under_test_dirs


pytestmark = pytest.mark.unit

FIXTURE_USERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "users"


def _training_corpus(tmp_path):
    other = tmp_path / "OtherDataset" / "users"
    other.parent.mkdir(parents=True)
    shutil.copytree(FIXTURE_USERS_DIR, other)
    return str(other)


def test_an_excluded_user_under_a_test_dir_is_refused_by_name(tmp_path):
    training = _training_corpus(tmp_path)
    excluded = str(FIXTURE_USERS_DIR / "1")
    with pytest.raises(ValueError, match=r"exclude_users names 1 user\(s\) under test_dirs \(1\)") as excinfo:
        create_dataloader_from_path(
            training, batch_size=4, device=torch.device("cpu"), is_train=True,
            test_dir=str(FIXTURE_USERS_DIR), sample_time=1, sample_rate=10, samples_per_user=4,
            exclude_users=[excluded], test_on_excluded=False, seed=1,
        )
    assert "exclude_users=[]" in str(excinfo.value)


def test_the_same_configuration_with_no_exclusions_loads_every_evaluation_user(tmp_path):
    training = _training_corpus(tmp_path)
    _, test_loader = create_dataloader_from_path(
        training, batch_size=4, device=torch.device("cpu"), is_train=True,
        test_dir=str(FIXTURE_USERS_DIR), sample_time=1, sample_rate=10, samples_per_user=4,
        exclude_users=[], test_on_excluded=False, seed=1,
    )
    fixture_users = sum(1 for p in FIXTURE_USERS_DIR.iterdir() if p.is_dir())
    assert test_loader.dataset.num_users == fixture_users == 2


def test_exclusions_under_the_training_corpus_are_still_allowed(tmp_path):
    """The legitimate use - carving validation or held-out users out of TRAINING - is untouched."""
    training = _training_corpus(tmp_path)
    excluded = str(pathlib.Path(training) / "1")
    _, test_loader = create_dataloader_from_path(
        training, batch_size=4, device=torch.device("cpu"), is_train=True,
        test_dir=str(FIXTURE_USERS_DIR), sample_time=1, sample_rate=10, samples_per_user=4,
        exclude_users=[excluded], test_on_excluded=False, seed=1,
    )
    assert test_loader.dataset.num_users == 2


def test_test_on_excluded_true_is_not_the_trap():
    """With the flag set, excluded users ARE the evaluation set by design; nothing to refuse."""
    refuse_excluded_users_under_test_dirs(str(FIXTURE_USERS_DIR), [])   # no exclusions: fine
    # The guard is only consulted when test_on_excluded is false, so a direct call with
    # offenders is the refusal itself:
    with pytest.raises(ValueError):
        refuse_excluded_users_under_test_dirs(str(FIXTURE_USERS_DIR), [str(FIXTURE_USERS_DIR / "2")])

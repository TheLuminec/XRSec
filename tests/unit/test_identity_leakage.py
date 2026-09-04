"""Evaluation users must be unseen unless seen-user evaluation is asked for."""
import types
from pathlib import Path

import pytest

from dataset import assert_evaluation_users_are_unseen

pytestmark = pytest.mark.unit


def _dataset(user_dirs):
    return types.SimpleNamespace(
        sample_index=types.SimpleNamespace(user_dirs=list(user_dirs)))


def test_disjoint_corpora_pass():
    """The new protocol: train on BOXRR, evaluate on the messy corpus."""
    train = _dataset(["/d/BOXRR/users/a", "/d/BOXRR/users/b"])
    test = _dataset(["/d/ViewGauss/users/1", "/d/NJIT/users/2"])
    assert assert_evaluation_users_are_unseen(train, test) == 0


def test_a_shared_user_is_refused():
    """
    The failure this exists for: a dataset left in both data_dirs and test_dirs, or a
    test_dirs that silently falls back to data_dirs. The number would be much higher
    and entirely plausible.
    """
    train = _dataset(["/d/A/users/1", "/d/A/users/2"])
    test = _dataset(["/d/A/users/2", "/d/B/users/9"])
    with pytest.raises(ValueError, match="also trained on"):
        assert_evaluation_users_are_unseen(train, test)


def test_the_error_names_the_offending_users():
    train = _dataset(["/d/A/users/alice"])
    test = _dataset(["/d/A/users/alice"])
    with pytest.raises(ValueError, match="alice"):
        assert_evaluation_users_are_unseen(train, test)


def test_seen_user_evaluation_is_allowed_when_asked_for(capsys):
    """
    Deliberate seen-user evaluation is how the historical 0.85 was reproduced and shown
    to be a seen-user figure. It stays available; it just has to be requested.
    """
    train = _dataset(["/d/A/users/1", "/d/A/users/2"])
    test = _dataset(["/d/A/users/1"])
    assert assert_evaluation_users_are_unseen(train, test, allow_seen_user_eval=True) == 1
    assert "SEEN users" in capsys.readouterr().out


def test_paths_are_compared_after_resolution():
    """A trailing slash or a relative form is the same user, and must not slip past."""
    train = _dataset(["/d/A/users/1"])
    test = _dataset(["/d/A/users/1/"])
    with pytest.raises(ValueError):
        assert_evaluation_users_are_unseen(train, test)


def test_a_dataset_with_no_recorded_users_is_not_a_failure():
    """Older or synthetic indexes record nothing; that is unknown, not a violation."""
    assert assert_evaluation_users_are_unseen(_dataset([]), _dataset(["/d/A/users/1"])) == 0


def test_wrapped_datasets_are_unwrapped():
    inner = _dataset(["/d/A/users/1"])
    wrapped = types.SimpleNamespace(dataset=inner)
    with pytest.raises(ValueError):
        assert_evaluation_users_are_unseen(wrapped, inner)

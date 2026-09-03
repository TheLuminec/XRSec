"""A checkpoint must say which users it never saw."""
from types import SimpleNamespace

import pytest

from eval import _resolve_eval_split

pytestmark = pytest.mark.unit


def _args(**overrides):
    args = SimpleNamespace(
        data_dirs=["/x/A/users"], test_dirs=[], exclude_users=["/x/A/users/1"],
        swap_data=False, test_on_excluded=True, sample_time=2, sample_rate=20,
        encoding="raw", resample="nearest", center_position=False,
        use_checkpoint_split=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _recorded(**overrides):
    split = {
        "data_dirs": ["/x/A/users", "/x/B/users"], "test_dirs": [],
        "exclude_users": [f"/x/A/users/{i}" for i in range(69)],
        "swap_data": False, "test_on_excluded": True,
        "sample_time": 2, "sample_rate": 20, "encoding": "raw",
        "resample": "nearest", "center_position": False,
    }
    split.update(overrides)
    return {"eval_split": split}


def test_the_recorded_split_is_used_over_the_config():
    """
    The defect this exists for: a sweep-fold checkpoint carried no record of its own
    fold, so evaluating it fell back to the config's default 5-user split - silently,
    and towards a split CLAUDE.md documents as unusually easy.
    """
    dirs, excluded, swap, on_excluded = _resolve_eval_split(_args(), _recorded())
    assert len(excluded) == 69, "fell back to the config's split"
    assert dirs == ["/x/A/users", "/x/B/users"]
    assert on_excluded is True


def test_a_checkpoint_without_a_split_warns_loudly(capsys):
    dirs, excluded, swap, on_excluded = _resolve_eval_split(_args(), {"history": {}})
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out
    assert "5 users" in out
    assert len(excluded) == 1          # the config's, used but flagged


def test_a_recovered_split_says_so(capsys):
    _resolve_eval_split(_args(), _recorded())
    assert "recovered from the checkpoint" in capsys.readouterr().out


def test_the_recorded_split_can_be_overridden_deliberately():
    """Cross-dataset evaluation is a real use; it just has to be asked for."""
    args = _args(use_checkpoint_split=False, test_dirs=["/x/C/users"])
    dirs, excluded, _, _ = _resolve_eval_split(args, _recorded())
    assert dirs == ["/x/C/users"]
    assert len(excluded) == 1


def test_windows_built_differently_than_at_training_time_warn(capsys):
    """A recovered user list is no use if the windows are not the ones it was trained on."""
    _resolve_eval_split(_args(sample_rate=10, encoding="bra"), _recorded())
    out = capsys.readouterr().out
    assert "sample_rate was 20 at training time, 10 now" in out
    assert "encoding was raw at training time, bra now" in out


def test_no_warning_when_everything_matches(capsys):
    _resolve_eval_split(_args(), _recorded())
    assert "WARNING" not in capsys.readouterr().out

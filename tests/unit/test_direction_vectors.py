"""A unit vector in the position column is a different quantity, not a convention."""
import types

import pytest
import torch

from dataset import detect_direction_vector_datasets

pytestmark = pytest.mark.unit


def _index(positions, dataset_ids=None, names=("DS",), channels=7):
    n, t = positions.shape[0], positions.shape[2]
    samples = torch.zeros(n, channels, t)
    where = slice(4, 7) if channels >= 7 else slice(0, 3)
    samples[:, where] = positions
    return types.SimpleNamespace(
        samples=samples, sample_count=n, num_channels=channels,
        dataset_names=list(names),
        window_dataset_ids=(dataset_ids if dataset_ids is not None
                            else torch.zeros(n, dtype=torch.long)))


def _unit(n=40, t=10, seed=0):
    torch.manual_seed(seed)
    v = torch.randn(n, 3, t)
    return v / v.norm(dim=1, keepdim=True)


def test_a_unit_direction_field_is_flagged():
    """PanoSaliency, Panonut360, Head_and_Gaze V1 and 360_em all look like this."""
    assert detect_direction_vector_datasets(_index(_unit())) == ["DS"]


def test_a_real_position_field_is_not_flagged():
    torch.manual_seed(1)
    positions = torch.randn(40, 3, 10) * 0.5 + torch.tensor([0.0, 1.6, 0.0]).view(1, 3, 1)
    assert detect_direction_vector_datasets(_index(positions)) == []


def test_only_the_offending_dataset_is_named():
    """Pooled corpora mix both kinds; the warning has to be specific."""
    torch.manual_seed(2)
    real = torch.randn(30, 3, 10) * 0.4 + torch.tensor([0.0, 1.6, 0.0]).view(1, 3, 1)
    positions = torch.cat([real, _unit(30)])
    ids = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    flagged = detect_direction_vector_datasets(
        _index(positions, ids, names=("RealPos", "DirectionOnly")))
    assert flagged == ["DirectionOnly"]


def test_a_constant_position_is_not_mistaken_for_a_direction():
    """Zero-variance is not the test; magnitude exactly 1 is."""
    positions = torch.full((30, 3, 10), 0.5)          # constant, norm 0.87
    assert detect_direction_vector_datasets(_index(positions)) == []


def test_the_warning_says_what_it_means(capsys):
    detect_direction_vector_datasets(_index(_unit()))
    out = capsys.readouterr().out
    assert "DIRECTION VECTOR" in out and "not a position" in out
    out.encode("cp1252")          # Windows consoles are cp1252


def test_the_position_only_channel_set_is_handled():
    assert detect_direction_vector_datasets(_index(_unit(), channels=3)) == ["DS"]


def test_an_empty_index_is_not_a_failure():
    empty = types.SimpleNamespace(samples=torch.empty(0, 7, 10), sample_count=0,
                                  num_channels=7, dataset_names=["DS"],
                                  window_dataset_ids=torch.empty(0, dtype=torch.long))
    assert detect_direction_vector_datasets(empty) == []

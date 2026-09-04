"""The training-free baseline, computed on every run."""
import types

import pytest
import torch

from metrics import per_dataset_metrics, static_position_lookup

pytestmark = pytest.mark.unit


def test_the_lookup_is_a_similarity_not_a_distance():
    """Higher must mean more alike, so it scores like any other head."""
    a = torch.tensor([[0.0, 1.6, 0.0], [0.0, 1.6, 0.0]])
    b = torch.tensor([[0.0, 1.6, 0.0], [0.0, 1.9, 0.0]])
    scores = static_position_lookup(a, b)
    assert scores[0] > scores[1]


def test_identical_positions_score_zero():
    a = torch.randn(8, 3)
    assert torch.allclose(static_position_lookup(a, a), torch.zeros(8), atol=1e-6)


def test_it_separates_users_who_differ_in_height():
    """
    Three numbers per window. On five folds this matched the trained model, so the
    property being pinned here is the one the whole comparison rests on.
    """
    torch.manual_seed(0)
    heights = torch.tensor([1.5, 1.7, 1.9])
    left, right, labels = [], [], []
    for i, h in enumerate(heights):
        for j, g in enumerate(heights):
            left.append(torch.tensor([0.0, h, 0.0]) + 0.01 * torch.randn(3))
            right.append(torch.tensor([0.0, g, 0.0]) + 0.01 * torch.randn(3))
            labels.append(1.0 if i == j else 0.0)
    from metrics import roc_auc
    scores = static_position_lookup(torch.stack(left), torch.stack(right))
    assert roc_auc(scores, torch.tensor(labels)) > 0.9


def test_per_dataset_metrics_split_by_corpus():
    """
    Pooling averages 0.93+ where real position exists with chance where the column holds
    a direction vector. The split is what makes either number readable.
    """
    torch.manual_seed(1)
    labels = torch.tensor([1.0, 0.0] * 10)
    good = torch.where(labels > 0, torch.rand(20) + 1.0, torch.rand(20))
    noise = torch.rand(20)
    scores = torch.cat([good, noise])
    ids = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    out = per_dataset_metrics(scores, torch.cat([labels, labels]), ids,
                              dataset_names=["RealPosition", "DirectionOnly"])
    assert out["RealPosition"]["auc"] > 0.9
    assert 0.3 < out["DirectionOnly"]["auc"] < 0.7
    assert out["RealPosition"]["pairs"] == 20


def test_a_single_class_dataset_is_skipped_not_reported_as_chance():
    """AUC is undefined without both classes; reporting 0.5 there would be a lie."""
    scores = torch.rand(10)
    labels = torch.ones(10)
    assert per_dataset_metrics(scores, labels, torch.zeros(10, dtype=torch.long)) == {}


def test_dataset_names_fall_back_to_ids():
    torch.manual_seed(2)
    labels = torch.tensor([1.0, 0.0] * 5)
    out = per_dataset_metrics(torch.rand(10), labels, torch.zeros(10, dtype=torch.long))
    assert "0" in out

import math

import pytest
import torch

from metrics import equal_error_rate, pair_metrics, roc_auc


pytestmark = pytest.mark.unit


def test_perfect_separation():
    scores = torch.tensor([-3.0, -2.0, 1.0, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    eer, _ = equal_error_rate(scores, labels)
    assert eer == pytest.approx(0.0)


def test_inverted_separation():
    scores = torch.tensor([3.0, 2.0, -1.0, -2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_all_tied_scores_give_chance_auc():
    """
    An untrained model outputs a near-constant logit. Without averaged tie ranks this
    silently reports 0.0 or 1.0 depending on sort order rather than 0.5.
    """
    scores = torch.zeros(100)
    labels = torch.cat([torch.ones(50), torch.zeros(50)])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_auc_matches_brute_force_pair_counting():
    torch.manual_seed(0)
    scores = torch.randn(60)
    labels = (torch.rand(60) > 0.5).float()

    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    expected = float(wins) / (positives.numel() * negatives.numel())

    assert roc_auc(scores, labels) == pytest.approx(expected, abs=1e-9)


def test_eer_is_balanced_at_its_threshold():
    torch.manual_seed(1)
    scores = torch.cat([torch.randn(200) + 1.5, torch.randn(200) - 1.5])
    labels = torch.cat([torch.ones(200), torch.zeros(200)])

    eer, threshold = equal_error_rate(scores, labels)
    false_accept = (scores[labels <= 0.5] >= threshold).float().mean().item()
    false_reject = (scores[labels > 0.5] < threshold).float().mean().item()

    assert abs(false_accept - false_reject) < 0.02
    assert 0.0 <= eer <= 0.5


def test_chance_scores_give_eer_near_half():
    torch.manual_seed(2)
    scores = torch.randn(500)
    labels = (torch.rand(500) > 0.5).float()
    eer, _ = equal_error_rate(scores, labels)
    assert eer == pytest.approx(0.5, abs=0.08)


def test_single_class_returns_nan_rather_than_a_misleading_number():
    scores = torch.randn(10)
    labels = torch.ones(10)
    assert math.isnan(roc_auc(scores, labels))
    assert math.isnan(equal_error_rate(scores, labels)[0])


def test_pair_metrics_reports_all_three():
    scores = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    metrics = pair_metrics(scores, labels)
    assert set(metrics) == {"auc", "eer", "eer_threshold"}
    assert metrics["auc"] == pytest.approx(1.0)


def test_auc_is_invariant_to_monotonic_rescaling():
    """Threshold-free: only the ranking matters, not calibration."""
    torch.manual_seed(3)
    scores = torch.randn(80)
    labels = (torch.rand(80) > 0.5).float()
    assert roc_auc(scores, labels) == pytest.approx(roc_auc(scores * 7.5 + 3.0, labels), abs=1e-9)

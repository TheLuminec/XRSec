"""The O(n log n) EER must return exactly what the per-threshold sweep returned."""
import pytest
import torch

from metrics import equal_error_rate


pytestmark = pytest.mark.unit


def _reference(scores, labels):
    """The original implementation: one vectorised comparison per distinct score."""
    positive_scores = scores[labels > 0.5]
    negative_scores = scores[labels <= 0.5]
    thresholds = torch.unique(scores)
    thresholds = torch.cat([thresholds.min().view(1) - 1.0, thresholds])
    best_gap, best_eer, best_threshold = float("inf"), float("nan"), float("nan")
    for threshold in thresholds.tolist():
        false_accept = (negative_scores >= threshold).double().mean().item()
        false_reject = (positive_scores < threshold).double().mean().item()
        gap = abs(false_accept - false_reject)
        if gap < best_gap:
            best_gap, best_eer, best_threshold = gap, (false_accept + false_reject) / 2.0, threshold
    return best_eer, best_threshold


@pytest.mark.parametrize("seed", range(6))
def test_matches_the_threshold_sweep_on_continuous_scores(seed):
    g = torch.Generator().manual_seed(seed)
    labels = (torch.rand(400, generator=g) > 0.5).float()
    scores = torch.randn(400, generator=g) + labels * 0.8
    eer, threshold = equal_error_rate(scores, labels)
    ref_eer, ref_threshold = _reference(scores, labels)
    assert eer == pytest.approx(ref_eer, abs=1e-9)
    assert threshold == pytest.approx(ref_threshold, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_matches_the_threshold_sweep_with_heavy_ties(seed):
    """Discrete scores make thresholds tie constantly; the first minimum must win."""
    g = torch.Generator().manual_seed(100 + seed)
    labels = (torch.rand(300, generator=g) > 0.5).float()
    scores = torch.randint(0, 6, (300,), generator=g).float() + labels
    eer, threshold = equal_error_rate(scores, labels)
    ref_eer, ref_threshold = _reference(scores, labels)
    assert eer == pytest.approx(ref_eer, abs=1e-9)
    assert threshold == pytest.approx(ref_threshold, abs=1e-6)


def test_perfect_and_constant_scorers():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    eer, _ = equal_error_rate(torch.tensor([2.0, 3.0, 0.0, 1.0]), labels)
    assert eer == pytest.approx(0.0)
    eer, _ = equal_error_rate(torch.zeros(4), labels)
    assert eer == pytest.approx(0.5)


def test_single_class_is_nan():
    eer, threshold = equal_error_rate(torch.rand(5), torch.ones(5))
    assert eer != eer and threshold != threshold

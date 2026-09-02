"""
Verification metrics for same/different pair decisions.

Accuracy alone is a weak summary here. It is measured at one fixed threshold (logit
> 0), so it conflates how well the model *ranks* pairs with whether its operating
point happens to be well placed, and a model can look flat at 0.50 while still
carrying usable signal. The standard biometric verification metrics separate those:

    AUC  probability that a random same-user pair scores above a random
         different-user pair. Threshold-free, so it measures ranking alone.
    EER  the error rate where false accepts equal false rejects, with the threshold
         that achieves it. This is the number biometrics papers report, and it is
         directly comparable across datasets with different pair balance.

Both are computed from raw logits, so no calibration is assumed.
"""

from __future__ import annotations

import torch


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Area under the ROC curve, via the rank-sum identity.

    Ties are handled by averaging their ranks, which matters for an untrained model
    whose outputs are nearly constant.
    """
    scores = scores.detach().float().view(-1).cpu()
    labels = labels.detach().float().view(-1).cpu()

    positives = int((labels > 0.5).sum())
    negatives = int(labels.numel() - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)

    # Average the ranks within each run of equal scores.
    start = 0
    for end in range(1, sorted_scores.numel() + 1):
        if end == sorted_scores.numel() or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                block = order[start:end]
                ranks[block] = ranks[block].mean()
            start = end

    positive_rank_sum = ranks[labels > 0.5].sum().item()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def equal_error_rate(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """
    Equal error rate and the threshold that achieves it.

    Sweeps every distinct score as a candidate threshold and returns the point where
    the false-accept and false-reject rates are closest.
    """
    scores = scores.detach().float().view(-1).cpu()
    labels = labels.detach().float().view(-1).cpu()

    positive_scores = scores[labels > 0.5]
    negative_scores = scores[labels <= 0.5]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return float("nan"), float("nan")

    thresholds = torch.unique(scores)
    # A single candidate per distinct score is enough; add one below the minimum so
    # the "accept everything" end of the curve is reachable.
    thresholds = torch.cat([thresholds.min().view(1) - 1.0, thresholds])

    best_gap = float("inf")
    best_eer = float("nan")
    best_threshold = float("nan")
    for threshold in thresholds.tolist():
        false_accept = (negative_scores >= threshold).float().mean().item()
        false_reject = (positive_scores < threshold).float().mean().item()
        gap = abs(false_accept - false_reject)
        if gap < best_gap:
            best_gap = gap
            best_eer = (false_accept + false_reject) / 2.0
            best_threshold = threshold
    return best_eer, best_threshold


def pair_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    """AUC, EER and the EER threshold for one set of pair scores."""
    eer, threshold = equal_error_rate(scores, labels)
    return {"auc": roc_auc(scores, labels), "eer": eer, "eer_threshold": threshold}

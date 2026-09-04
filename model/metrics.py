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


def per_dataset_metrics(scores, labels, dataset_ids, dataset_names=None) -> dict:
    """
    AUC and EER split by the dataset each pair came from.

    A pooled number here is an average over corpora that differ in KIND, not merely in
    difficulty: measured held-out AUC runs 0.93+ where a real head position exists and
    0.49-0.58 where the position column holds a unit direction vector instead. Reporting
    only the pooled figure averages near-perfect verification with chance and hides both.
    """
    import torch as _torch

    out = {}
    ids = _torch.as_tensor(dataset_ids)
    for dataset_id in sorted(set(int(i) for i in ids.tolist())):
        mask = ids == dataset_id
        if int(mask.sum()) < 2:
            continue
        subset_labels = labels[mask]
        # AUC is undefined without both classes present.
        if float(subset_labels.min()) == float(subset_labels.max()):
            continue
        name = (dataset_names[dataset_id]
                if dataset_names and dataset_id < len(dataset_names) else str(dataset_id))
        metrics = pair_metrics(scores[mask], subset_labels)
        out[name] = {"auc": metrics["auc"], "eer": metrics["eer"],
                     "pairs": int(mask.sum())}
    return out


def static_position_lookup(left_positions, right_positions):
    """
    The training-free baseline: distance between two windows' MEAN POSITION.

    Three numbers per window, no model. Measured on five folds against the same held-out
    users and the same pair manifests, this scores pooled AUC 0.726 where the trained
    model scores 0.723 - and on an unseen dataset it transfers BETTER (0.593 against
    0.566). It has now been found competitive twice, both times only because someone went
    looking, so it is computed on every run rather than left as an occasional probe.

    Returns a similarity, so higher means more alike and it can be scored like any other.
    """
    return -(left_positions - right_positions).norm(dim=1)

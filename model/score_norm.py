"""
Adaptive score normalization (AS-Norm) for verification scores.

Accuracy here is read at a fixed threshold (`logit > 0`), which assumes one operating
point serves every user. It does not. Some identities sit in a dense part of the
embedding space and score high against everyone; others sit alone and score low even
against themselves. A single global threshold is then wrong in opposite directions for
the two, and the fixed-threshold accuracy this project reports is exactly the metric
that suffers - it has already failed twice for related reasons (label imbalance, and
selection on the reported set).

AS-Norm rescales each score by how surprising it is *for the two sides involved*,
using a cohort of impostors:

    z = 0.5 * [ (s - mean(top_k(s_left_vs_cohort))) / sd(...)
              + (s - mean(top_k(s_right_vs_cohort))) / sd(...) ]

Taking the top k rather than the whole cohort is what makes it "adaptive": the
informative impostors are the ones that come close, and averaging over a large cohort
of obviously-different identities washes that out.

**The cohort must be training identities.** Drawing it from the evaluation users would
let the test set shape its own normalization - the same mistake `ChannelNormalizer`
exists to avoid, and the same shape as every other leak this project has had to remove.
`cohort_from_training_users` is the only constructor offered here for that reason.

This is standard practice in speaker verification, where it is worth a few percent
relative EER for no retraining. It needs no gradient, no new data, and no change to the
model - only embeddings that have already been computed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_COHORT_SIZE = 400
DEFAULT_TOP_K = 200


def cohort_from_training_users(embeddings: torch.Tensor, sample_index,
                               exclude_users: set[int] | None = None,
                               per_user: int = 4, size: int = DEFAULT_COHORT_SIZE,
                               seed: int = 67, normalise: bool = True) -> torch.Tensor:
    """
    Build an impostor cohort by taking a few windows from each available identity.

    Sampling per user rather than uniformly over windows matters for the same reason
    balanced identity sampling does: window counts span nearly 90x, so a uniform draw
    would build a cohort dominated by a handful of well-recorded people and the
    normalization statistics would describe them rather than the population.
    """
    generator = torch.Generator().manual_seed(int(seed))
    exclude_users = exclude_users or set()

    picked = []
    for user_index, windows in enumerate(sample_index.user_sample_indices):
        if user_index in exclude_users or windows.numel() == 0:
            continue
        take = min(per_user, int(windows.numel()))
        order = torch.randperm(int(windows.numel()), generator=generator)[:take]
        picked.append(windows[order])

    if not picked:
        return torch.empty(0)

    indices = torch.cat(picked)
    if indices.numel() > size:
        order = torch.randperm(int(indices.numel()), generator=generator)[:size]
        indices = indices[order]

    cohort = embeddings[indices]
    return F.normalize(cohort, dim=1) if normalise else cohort


def _cohort_statistics(vectors: torch.Tensor, cohort: torch.Tensor,
                       top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and sd of each vector's top-k similarities against the cohort."""
    similarities = vectors @ cohort.T                     # (n, cohort)
    top_k = max(2, min(int(top_k), similarities.shape[1]))
    top = similarities.topk(top_k, dim=1).values
    # Unbiased sd needs at least two values; the clamp keeps a degenerate cohort from
    # producing a divide-by-zero that would silently become inf scores.
    return top.mean(dim=1), top.std(dim=1).clamp(min=1e-6)


def as_norm(scores: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
            cohort: torch.Tensor, top_k: int = DEFAULT_TOP_K,
            normalise: bool = True) -> torch.Tensor:
    """
    Normalize each score against how the two sides score on an impostor cohort.

    `left` and `right` are the embeddings the score was computed from, one row per
    score. Returns a tensor of the same shape; ordering, not magnitude, is what the
    result is for, so a threshold fitted on raw scores does not transfer.
    """
    if scores.numel() == 0 or cohort.numel() == 0:
        return scores

    if normalise:
        left = F.normalize(left, dim=1)
        right = F.normalize(right, dim=1)

    left_mean, left_sd = _cohort_statistics(left, cohort, top_k)
    right_mean, right_sd = _cohort_statistics(right, cohort, top_k)

    return 0.5 * ((scores - left_mean) / left_sd + (scores - right_mean) / right_sd)

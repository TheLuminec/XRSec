"""Adaptive score normalization."""
import types

import pytest
import torch
import torch.nn.functional as F

from score_norm import as_norm, cohort_from_training_users

pytestmark = pytest.mark.unit


def _index(users=6, per_user=8, dim=16, seed=0):
    torch.manual_seed(seed)
    indices, vectors = [], []
    for user in range(users):
        centre = torch.randn(dim)
        indices.append(torch.arange(user * per_user, (user + 1) * per_user))
        vectors.append(centre + 0.1 * torch.randn(per_user, dim))
    return (types.SimpleNamespace(user_sample_indices=indices,
                                  sample_count=users * per_user, num_users=users),
            torch.cat(vectors))


def test_the_cohort_takes_a_few_windows_from_every_identity():
    """
    Window counts span ~90x, so a uniform draw would build a cohort describing a
    handful of well-recorded people rather than the population.
    """
    index, embeddings = _index()
    index.user_sample_indices[0] = torch.arange(0, 8).repeat(20)   # one dominant user
    cohort = cohort_from_training_users(embeddings, index, per_user=2, size=1000)
    assert cohort.shape[0] == 12       # 6 users x 2, not dominated by user 0


def test_evaluation_users_can_be_kept_out_of_the_cohort():
    """
    A cohort drawn from the evaluation users would let the test set shape its own
    normalization - the same leak ChannelNormalizer exists to prevent.
    """
    index, embeddings = _index()
    cohort = cohort_from_training_users(embeddings, index, exclude_users={0, 1},
                                        per_user=3, size=1000)
    assert cohort.shape[0] == 12       # 4 remaining users x 3


def test_cohort_rows_are_unit_norm_when_normalising():
    index, embeddings = _index()
    cohort = cohort_from_training_users(embeddings, index, per_user=2)
    assert torch.allclose(cohort.norm(dim=1), torch.ones(cohort.shape[0]), atol=1e-5)


def test_normalisation_is_deterministic_for_a_seed():
    index, embeddings = _index()
    a = cohort_from_training_users(embeddings, index, per_user=2, seed=5)
    b = cohort_from_training_users(embeddings, index, per_user=2, seed=5)
    assert torch.equal(a, b)


def test_a_side_that_scores_high_against_everyone_is_discounted():
    """
    The point of the method: a score of 0.9 from someone who scores 0.9 against the
    whole cohort is not evidence, and a global threshold cannot tell the difference.
    """
    dim = 8
    cohort = F.normalize(torch.randn(64, dim), dim=1)
    crowded = cohort[:1].repeat(1, 1)             # sits exactly on the cohort
    lonely = F.normalize(-cohort.mean(dim=0, keepdim=True), dim=1)

    left = torch.cat([crowded, lonely])
    right = left.clone()
    raw = torch.tensor([0.9, 0.9])

    normalised = as_norm(raw, left, right, cohort, top_k=16)
    assert normalised[1] > normalised[0], "the unusual match should survive better"


def test_ordering_changes_but_shape_does_not():
    index, embeddings = _index()
    cohort = cohort_from_training_users(embeddings, index, per_user=3)
    left, right = embeddings[:20], embeddings[20:40]
    raw = F.cosine_similarity(left, right)
    out = as_norm(raw, left, right, cohort, top_k=8)
    assert out.shape == raw.shape
    assert torch.isfinite(out).all()


def test_an_empty_cohort_returns_the_scores_untouched():
    """No cohort is a reason to report the raw number, not to emit NaNs."""
    raw = torch.tensor([0.4, 0.7])
    out = as_norm(raw, torch.randn(2, 8), torch.randn(2, 8), torch.empty(0))
    assert torch.equal(out, raw)


def test_a_degenerate_cohort_does_not_produce_infinities():
    """Identical cohort rows give zero spread; that must clamp, not divide by zero."""
    cohort = F.normalize(torch.ones(32, 8), dim=1)
    out = as_norm(torch.tensor([0.5, 0.5]), torch.randn(2, 8), torch.randn(2, 8),
                  cohort, top_k=8)
    assert torch.isfinite(out).all()

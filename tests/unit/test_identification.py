"""Closed-set identification metrics."""
import types

import numpy as np
import pytest
import torch

from templates import cmc_curve, format_cmc

pytestmark = pytest.mark.unit


class _CosineModel:
    """Scores the way an identity_softmax model does."""
    head = "cosine"

    def score(self, left, right):
        return (left * right).sum(dim=1)


def _index(num_users=6, sessions_per_user=2, windows_per_session=8, dim=6):
    """Sample index plus embeddings where each user occupies its own direction."""
    per_session = windows_per_session
    total_per_user = sessions_per_user * per_session

    user_sample_indices, sessions, vectors = [], [], []
    for user in range(num_users):
        base = user * total_per_user
        user_sample_indices.append(torch.arange(base, base + total_per_user))
        sessions.append(torch.arange(total_per_user) // per_session)
        direction = torch.zeros(dim)
        direction[user % dim] = 1.0
        vectors.append(direction.repeat(total_per_user, 1))

    index = types.SimpleNamespace(
        num_users=num_users,
        sample_count=num_users * total_per_user,
        user_sample_indices=user_sample_indices,
        window_session_ids=torch.cat(sessions),
        sample_time=2,
    )
    return index, torch.cat(vectors)


def test_a_separable_embedding_identifies_everyone():
    index, embeddings = _index()
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=5, seed=1)
    assert result["users"] == 6
    assert result["rank1"] == 1.0
    assert result["chance"] == pytest.approx(1 / 6)


def test_a_constant_embedding_lands_at_the_middle_of_the_gallery():
    """
    Every score identical is no information. Breaking those ties by sort order would
    report rank 1 or rank N; rank-averaging reports (N+1)/2, which is what it is.
    """
    index, embeddings = _index()
    constant = torch.ones_like(embeddings)
    result = cmc_curve(_CosineModel(), constant, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=5, seed=1)
    assert result["rank1"] == 0.0
    assert result["mean_rank"] == pytest.approx((result["users"] + 1) / 2)


def test_chance_is_one_over_the_gallery_size_not_one_half():
    """
    The whole point of adding this: verification chance is 0.50, identification chance
    is 1/N, and a rank-1 quoted without its N cannot be compared to anything.
    """
    for num_users in (4, 10):
        index, embeddings = _index(num_users=num_users)
        result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                           gallery_k=4, probe_k=1, probes_per_user=3, seed=2)
        assert result["chance"] == pytest.approx(1 / num_users)
        assert result["users"] == num_users


def test_the_cmc_curve_never_decreases():
    index, embeddings = _index()
    noisy = embeddings + 0.5 * torch.randn_like(embeddings)
    result = cmc_curve(_CosineModel(), noisy, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=8, seed=3)
    cmc = result["cmc"]
    assert len(cmc) == result["users"]
    assert all(b >= a - 1e-9 for a, b in zip(cmc, cmc[1:]))
    assert cmc[-1] == pytest.approx(1.0)


def test_gallery_and_probe_come_from_different_sessions():
    """A match found within one session would be session matching, not identification."""
    index, embeddings = _index(sessions_per_user=2, windows_per_session=8)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=5, seed=4)
    assert result["same_session_fallback_users"] == 0


def test_single_session_users_are_counted_not_hidden():
    index, embeddings = _index(sessions_per_user=1, windows_per_session=16)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=5, seed=5)
    assert result["same_session_fallback_users"] == result["users"]
    assert "one session" in format_cmc(result)


def test_users_with_too_few_windows_are_skipped_and_reported():
    index, embeddings = _index(num_users=4, sessions_per_user=1, windows_per_session=2)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=8, probe_k=1, probes_per_user=3, seed=6)
    assert result["skipped_users"] == 4
    assert result["users"] == 0


def test_a_larger_gallery_template_is_accepted():
    index, embeddings = _index(windows_per_session=8)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=8, probe_k=2, probes_per_user=4, seed=7)
    assert result["gallery_k"] == 8 and result["probe_k"] == 2
    assert result["rank1"] == 1.0


def test_format_is_ascii_only():
    """Windows consoles are cp1252; box drawing crashes as soon as stdout is piped."""
    index, embeddings = _index()
    text = format_cmc(cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                                gallery_k=4, probe_k=1, probes_per_user=3, seed=8))
    text.encode("cp1252")
    assert "rank-1" in text and "chance" in text

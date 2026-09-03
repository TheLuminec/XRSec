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


# --- matched gallery size -----------------------------------------------------
#
# An external rank-1 measured over 17 users is not comparable to one measured over
# 419: chance is 1/N, and ranking against fewer candidates is an easier problem.

def test_rank1_is_reported_at_a_restricted_gallery_size():
    index, embeddings = _index(num_users=10)
    noisy = embeddings + 0.8 * torch.randn_like(embeddings)
    result = cmc_curve(_CosineModel(), noisy, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=8, seed=1,
                       gallery_sizes=(3, 5), subsets=10)

    matched = result["rank1_at_gallery_size"]
    assert set(matched) == {3, 5}
    assert matched[3]["chance"] == pytest.approx(1 / 3)
    assert matched[5]["chance"] == pytest.approx(1 / 5)


def test_a_smaller_gallery_is_an_easier_problem():
    """The whole reason matched N matters: fewer candidates means higher rank-1."""
    index, embeddings = _index(num_users=12)
    noisy = embeddings + 1.2 * torch.randn_like(embeddings)
    result = cmc_curve(_CosineModel(), noisy, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=12, seed=2,
                       gallery_sizes=(2, 4, 12), subsets=20)

    matched = result["rank1_at_gallery_size"]
    assert matched[2]["rank1"] >= matched[4]["rank1"] >= matched[12]["rank1"] - 1e-9


def test_the_full_gallery_size_agrees_with_the_headline_rank1():
    index, embeddings = _index(num_users=8)
    noisy = embeddings + 0.6 * torch.randn_like(embeddings)
    result = cmc_curve(_CosineModel(), noisy, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=10, seed=3,
                       gallery_sizes=(8,), subsets=5)
    # N equal to the whole gallery is the same measurement, whatever subset is drawn.
    assert result["rank1_at_gallery_size"][8]["rank1"] == pytest.approx(result["rank1"])
    assert result["rank1_at_gallery_size"][8]["sd"] == pytest.approx(0.0, abs=1e-9)


def test_impossible_gallery_sizes_are_ignored_not_fatal():
    index, embeddings = _index(num_users=5)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=4, seed=4,
                       gallery_sizes=(1, 5, 50), subsets=5)
    assert set(result["rank1_at_gallery_size"]) == {5}


def test_matched_sizes_appear_in_the_formatted_output():
    index, embeddings = _index(num_users=6)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=4, seed=5,
                       gallery_sizes=(3,), subsets=5)
    text = format_cmc(result)
    text.encode("cp1252")
    assert "N=3" in text


def test_a_gallery_size_larger_than_the_gallery_is_reported(capsys):
    """
    Requesting [17,48,100] against a 5-user evaluation set used to return nothing at
    all, with no indication that the request had been dropped rather than measured.
    """
    index, embeddings = _index(num_users=5)
    result = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=4, seed=1,
                       gallery_sizes=(17, 48), subsets=3)
    assert result["rank1_at_gallery_size"] == {}
    assert "exceed the 5 enrolled users" in capsys.readouterr().out


# --- single-session users inflate rank-1 ---------------------------------------

def test_single_session_users_can_be_excluded_rather_than_fallen_back_on():
    """
    A single-session user's gallery and probe come from one recording, so a correct
    match may be session matching. On real folds that was 10-12 of ~62 users - ~17% of
    the gallery - inflating rank-1 by an unknown amount.
    """
    index, embeddings = _index(num_users=6, sessions_per_user=1, windows_per_session=16)
    lenient = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                        gallery_k=4, probe_k=1, probes_per_user=4, seed=1)
    strict = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=4, seed=1,
                       require_cross_session=True)
    assert lenient["users"] == 6
    assert strict["users"] == 0, "no user has two sessions, so none should enrol"


def test_cross_session_users_are_unaffected_by_the_flag():
    index, embeddings = _index(num_users=6, sessions_per_user=2, windows_per_session=8)
    lenient = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                        gallery_k=4, probe_k=1, probes_per_user=4, seed=2)
    strict = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                       gallery_k=4, probe_k=1, probes_per_user=4, seed=2,
                       require_cross_session=True)
    assert lenient["users"] == strict["users"] == 6
    assert strict["rank1"] == pytest.approx(lenient["rank1"])


def test_the_output_says_which_regime_produced_the_number(capsys):
    index, embeddings = _index(num_users=6, sessions_per_user=1, windows_per_session=16)
    lenient = cmc_curve(_CosineModel(), embeddings, index, torch.device("cpu"),
                        gallery_k=4, probe_k=1, probes_per_user=4, seed=3)
    assert "inflates rank-1" in format_cmc(lenient)

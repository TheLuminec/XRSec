import types

import pytest
import torch

from templates import (
    embed_all,
    format_curve,
    format_decomposition,
    variance_decomposition,
    generate_template_manifest,
    score_templates,
    window_curve,
)


pytestmark = pytest.mark.unit


def _index(users=4, sessions=3, per_session=8, seq_len=10, channels=7):
    """Sample index with session provenance, enough windows to build templates."""
    per_user = sessions * per_session
    samples = torch.randn(users * per_user, channels, seq_len)
    user_sample_indices, session_ids = [], []
    for user in range(users):
        start = user * per_user
        user_sample_indices.append(torch.arange(start, start + per_user, dtype=torch.long))
        session_ids.append(torch.arange(per_user) // per_session)
    return types.SimpleNamespace(
        samples=samples,
        sample_count=samples.shape[0],
        num_users=users,
        num_channels=channels,
        seq_len=seq_len,
        user_sample_indices=user_sample_indices,
        window_session_ids=torch.cat(session_ids),
        user_dataset_ids=[0] * users,
    )


def _session_of(index):
    return {i: int(index.window_session_ids[i]) for i in range(index.sample_count)}


def _user_of(index):
    return {i: u for u, ix in enumerate(index.user_sample_indices) for i in ix.tolist()}


# --- manifest construction ----------------------------------------------------

def test_each_side_holds_k_windows():
    index = _index()
    manifest = generate_template_manifest(index, pairs_per_user=8, k=4, seed=1)
    assert manifest["x1_indices"].shape[1] == 4
    assert manifest["x2_indices"].shape[1] == 4
    assert manifest["x1_indices"].shape[0] == manifest["labels"].shape[0]


def test_a_template_is_drawn_from_one_session():
    """
    Mixing sessions inside one template would average away the session variability
    that cross-session evaluation exists to expose.
    """
    index = _index()
    session = _session_of(index)
    manifest = generate_template_manifest(index, pairs_per_user=12, k=4, seed=2)

    for side in ("x1_indices", "x2_indices"):
        for row in manifest[side].tolist():
            assert len({session[i] for i in row}) == 1, "template spans two sessions"


def test_positive_sides_come_from_different_sessions_of_the_same_user():
    index = _index()
    session, user = _session_of(index), _user_of(index)
    manifest = generate_template_manifest(index, pairs_per_user=12, k=3, seed=3)

    positives = manifest["labels"] == 1
    assert int(positives.sum()) > 0
    for left, right in zip(manifest["x1_indices"][positives].tolist(),
                           manifest["x2_indices"][positives].tolist()):
        assert user[left[0]] == user[right[0]]
        assert session[left[0]] != session[right[0]]


def test_negative_sides_come_from_different_users():
    index = _index()
    user = _user_of(index)
    manifest = generate_template_manifest(index, pairs_per_user=12, k=3, seed=4)

    negatives = manifest["labels"] == 0
    assert int(negatives.sum()) > 0
    for left, right in zip(manifest["x1_indices"][negatives].tolist(),
                           manifest["x2_indices"][negatives].tolist()):
        assert user[left[0]] != user[right[0]]


def test_windows_within_a_template_are_distinct():
    index = _index()
    manifest = generate_template_manifest(index, pairs_per_user=8, k=4, seed=5)
    for side in ("x1_indices", "x2_indices"):
        for row in manifest[side].tolist():
            assert len(set(row)) == len(row), "a window was reused inside one template"


def test_sessions_too_short_for_k_are_skipped_and_counted():
    """
    Silently building a template from fewer than k windows would make the curve
    report a k it did not use.
    """
    index = _index(users=3, sessions=2, per_session=4)
    manifest = generate_template_manifest(index, pairs_per_user=4, k=8, seed=6)
    assert manifest["labels"].numel() == 0
    assert manifest["skipped_users"] == 3


def test_within_dataset_negatives_is_honoured():
    index = _index(users=4)
    index.user_dataset_ids = [0, 0, 1, 1]
    user = _user_of(index)
    manifest = generate_template_manifest(index, pairs_per_user=12, k=3, seed=7,
                                          within_dataset_negatives=True)
    negatives = manifest["labels"] == 0
    for left, right in zip(manifest["x1_indices"][negatives].tolist(),
                           manifest["x2_indices"][negatives].tolist()):
        assert index.user_dataset_ids[user[left[0]]] == index.user_dataset_ids[user[right[0]]]


def test_manifest_is_balanced():
    index = _index()
    manifest = generate_template_manifest(index, pairs_per_user=16, k=3, match_ratio=0.5, seed=8)
    assert float(manifest["labels"].mean()) == pytest.approx(0.5, abs=0.01)


def test_manifest_is_deterministic():
    index = _index()
    a = generate_template_manifest(index, pairs_per_user=8, k=3, seed=9)
    b = generate_template_manifest(index, pairs_per_user=8, k=3, seed=9)
    for key in ("x1_indices", "x2_indices", "labels"):
        assert torch.equal(a[key], b[key])


# --- scoring ------------------------------------------------------------------

def _model(head="cosine", embedding_dim=8, seq_len=10, channels=7):
    from model import create_model
    model, _, _ = create_model(embedding_dim=embedding_dim, seq_len=seq_len,
                               device=torch.device("cpu"), extractor="bilstm",
                               head=head, num_channels=channels)
    return model


def test_k1_reproduces_the_single_window_operating_point():
    """
    k=1 must be the existing measurement, or the curve is a replacement metric rather
    than an added dimension.
    """
    index = _index()
    model = _model()
    embeddings = embed_all(model, index, torch.device("cpu"))

    manifest = generate_template_manifest(index, pairs_per_user=8, k=1, seed=10)
    template_scores = score_templates(model, embeddings, manifest, torch.device("cpu"))

    left = manifest["x1_indices"].view(-1)
    right = manifest["x2_indices"].view(-1)
    model.eval()
    with torch.no_grad():
        direct = model(index.samples[left], index.samples[right]).view(-1)

    assert torch.allclose(template_scores, direct, atol=1e-5)


def test_averaging_reduces_the_spread_of_scores():
    """
    The premise of the whole idea: averaging k noisy observations of a stable
    per-person quantity should reduce variance in the score.
    """
    index = _index(users=6, sessions=3, per_session=16)
    model = _model()
    embeddings = embed_all(model, index, torch.device("cpu"))

    spreads = []
    for k in (1, 8):
        manifest = generate_template_manifest(index, pairs_per_user=32, k=k, seed=11)
        scores = score_templates(model, embeddings, manifest, torch.device("cpu"))
        positives = manifest["labels"] == 1
        spreads.append(float(scores[positives].std()))

    assert spreads[1] < spreads[0], "aggregating 8 windows did not reduce score spread"


def test_cosine_head_scoring_ignores_template_magnitude():
    model = _model(head="cosine")
    left, right = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(model.score(left, right), model.score(left * 4, right * 0.3), atol=1e-5)


def test_embeddings_are_computed_once_for_the_whole_curve(monkeypatch):
    """The cost claim: one forward pass serves every k, so no retraining is needed."""
    index = _index()
    model = _model()
    calls = []
    original = model.embed

    def counting(x):
        calls.append(x.shape[0])
        return original(x)

    monkeypatch.setattr(model, "embed", counting)
    window_curve(model, index, torch.device("cpu"), ks=(1, 2, 4),
                 pairs_per_user=8, seed=12, batch_size=1024)

    assert sum(calls) == index.sample_count, "windows were embedded more than once"


# --- curve --------------------------------------------------------------------

def test_curve_reports_a_row_per_k():
    index = _index(users=5, sessions=3, per_session=12)
    model = _model()
    rows = window_curve(model, index, torch.device("cpu"), ks=(1, 2, 4),
                        pairs_per_user=16, seed=13)

    assert [row["k"] for row in rows] == [1, 2, 4]
    for row in rows:
        assert 0.0 <= row["auc"] <= 1.0
        assert 0.0 <= row["eer"] <= 1.0
        assert row["positive_fraction"] == pytest.approx(0.5, abs=0.02)


def test_curve_notes_a_k_that_cannot_be_built():
    index = _index(users=3, sessions=2, per_session=4)
    model = _model()
    rows = window_curve(model, index, torch.device("cpu"), ks=(2, 32),
                        pairs_per_user=8, seed=14)
    assert rows[-1]["pairs"] == 0 and "note" in rows[-1]
    assert "32" in format_curve(rows)


# --- positive control ---------------------------------------------------------

def test_aggregation_demonstrably_improves_a_signal_it_should_improve():
    """
    Positive control for the k-sweep itself.

    A flat curve on real data is ambiguous: it is equally consistent with the
    structural bound (templates come from one session, so averaging cannot touch the
    between-session shift) and with a k-sweep that simply does not aggregate. This
    pins down the second possibility. Here each user has a constant per-identity
    offset plus independent per-window noise - exactly the case averaging must help -
    so if AUC does not climb with k, the machinery is broken and any flat result on
    real data means nothing.
    """
    torch.manual_seed(0)
    users, sessions, per_session, dim = 8, 3, 24, 16

    # Per-user signal, drowned in per-window noise that averaging can cancel.
    identity = torch.randn(users, dim) * 1.0
    embeddings = []
    for user in range(users):
        noise = torch.randn(sessions * per_session, dim) * 6.0
        embeddings.append(identity[user].unsqueeze(0) + noise)
    embeddings = torch.cat(embeddings)

    index = _index(users=users, sessions=sessions, per_session=per_session)

    class ConstantEmbedder(torch.nn.Module):
        head = "cosine"

        def score(self, left, right):
            return torch.nn.functional.cosine_similarity(left, right, dim=1).unsqueeze(1)

    model = ConstantEmbedder()

    aucs = []
    for k in (1, 8):
        manifest = generate_template_manifest(index, pairs_per_user=48, k=k, seed=21)
        scores = score_templates(model, embeddings, manifest, torch.device("cpu"))
        from metrics import roc_auc
        aucs.append(roc_auc(scores, manifest["labels"]))

    assert aucs[1] > aucs[0] + 0.05, (
        f"aggregation did not improve a signal it must improve (k=1 {aucs[0]:.3f}, "
        f"k=8 {aucs[1]:.3f}); a flat curve on real data cannot be interpreted while "
        "this fails"
    )


# --- variance decomposition ---------------------------------------------------

def _synthetic_embeddings(users, sessions, per_session, dim,
                          user_scale, session_scale, window_scale, seed=0):
    """Embeddings with a known three-level structure."""
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for user in range(users):
        identity = torch.randn(dim, generator=generator) * user_scale
        for _ in range(sessions):
            shift = torch.randn(dim, generator=generator) * session_scale
            noise = torch.randn(per_session, dim, generator=generator) * window_scale
            blocks.append(identity + shift + noise)
    return torch.cat(blocks)


def test_decomposition_separates_the_three_levels():
    """
    Averaging k windows can only reduce the within-session term. Measuring the split
    is what turns a flat curve from an argument into a prediction, so the measurement
    itself has to be right.
    """
    users, sessions, per_session, dim = 10, 4, 40, 12
    embeddings = _synthetic_embeddings(users, sessions, per_session, dim,
                                       user_scale=1.0, session_scale=0.3, window_scale=0.1)
    index = _index(users=users, sessions=sessions, per_session=per_session)

    parts = variance_decomposition(embeddings, index, normalise=False)

    # Ordering is what matters: signal above session shift above window noise.
    assert parts["between_user"] > parts["between_session"] > parts["within_session"]
    assert parts["signal_to_shift"] > 1.0


def test_decomposition_tracks_a_growing_session_shift():
    users, sessions, per_session, dim = 8, 4, 40, 12
    index = _index(users=users, sessions=sessions, per_session=per_session)

    small = variance_decomposition(
        _synthetic_embeddings(users, sessions, per_session, dim, 1.0, 0.1, 0.2, seed=1),
        index, normalise=False)
    large = variance_decomposition(
        _synthetic_embeddings(users, sessions, per_session, dim, 1.0, 0.9, 0.2, seed=1),
        index, normalise=False)

    assert large["between_session"] > small["between_session"]
    assert large["signal_to_shift"] < small["signal_to_shift"]


def test_plateau_estimate_moves_with_the_noise_ratio():
    """
    plateau_k is where within-session noise falls below the between-session shift.
    More window noise relative to session shift should push the plateau later.
    """
    users, sessions, per_session, dim = 8, 4, 60, 12
    index = _index(users=users, sessions=sessions, per_session=per_session)

    noisy = variance_decomposition(
        _synthetic_embeddings(users, sessions, per_session, dim, 1.0, 0.2, 1.0, seed=2),
        index, normalise=False)
    clean = variance_decomposition(
        _synthetic_embeddings(users, sessions, per_session, dim, 1.0, 0.2, 0.2, seed=2),
        index, normalise=False)

    assert noisy["plateau_k"] > clean["plateau_k"]


def test_decomposition_is_absent_without_session_provenance():
    index = _index()
    index.window_session_ids = torch.empty(0, dtype=torch.long)
    assert variance_decomposition(torch.randn(index.sample_count, 8), index) == {}
    assert "unavailable" in format_decomposition({})

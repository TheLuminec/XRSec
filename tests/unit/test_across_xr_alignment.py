"""
Fixture gate for docs/acceptance/across_xr_alignment.py.

The harness fits an orthogonal alignment between application embedding spaces on some users
and applies it to others. Before it touches a real embedding it has to get a known answer
right, and the fixture has to be asserted on - a test whose subject failed to load reports
the subject's success. Three fixtures with known answers:

  low-rank      user centroids live in a k-dim subspace of the 128-d space; application B is
                A rotated by a random orthogonal Q. The unaligned score must be near chance,
                the test-fitted ceiling near 1, and the TRAIN-fitted subspace alignment near 1
                too, because the fitting users' PCA captures the subspace the test users
                also live in. The permuted-correspondence null must stay near chance.
  isotropic     the same with centroids drawn isotropically in 128-d. Now 32 fitting users
                cannot span where the 17 test users live, and the train-fitted alignment
                must FAIL while the test-fitted one still succeeds - the failure mode the
                registration names, demonstrated rather than argued.
  rank          32 correspondences in 128-d: two orthonormal completions of the unrestricted
                Procrustes solve the same objective exactly and send an off-span vector to
                different places. That is why the subspace restriction is A2's definition.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

HARNESS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "acceptance"
sys.path.insert(0, str(HARNESS))

import across_xr_alignment as ax  # noqa: E402

D = 128
N_USERS, N_TEST = 49, 17
WINDOWS = 40


def _random_orthogonal(rng, d=D):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q = q * np.sign(np.diag(r))
    assert np.allclose(q.T @ q, np.eye(d), atol=1e-10), "fixture rotation is not orthogonal"
    return q


def _make_fixture(rng, k: int | None, noise: float = 0.15):
    """Per-user centroids (k-dim subspace if k, else isotropic), windows = centroid + noise,
    application B = application A @ Q. Returns embeddings, rows per (user, app), Q."""
    if k is None:
        cent = rng.standard_normal((N_USERS, D))
    else:
        basis = np.linalg.qr(rng.standard_normal((D, k)))[0]
        cent = rng.standard_normal((N_USERS, k)) @ basis.T
    cent = ax.l2(cent)
    q = _random_orthogonal(rng)
    emb, rows, offset = [], {}, 0
    for u in range(N_USERS):
        for app in ("A", "B"):
            w = cent[u] + noise * rng.standard_normal((WINDOWS, D))
            emb.append(w if app == "A" else w @ q)
            rows[(u, app)] = np.arange(offset, offset + WINDOWS)
            offset += WINDOWS
    return np.concatenate(emb), rows, q


def _score(emb, rows, users, r):
    gallery = ax.centroids(emb, [rows[(u, "A")] for u in users])
    probe_rows = np.concatenate([rows[(u, "B")] for u in users])
    probe_user = np.concatenate([np.full(WINDOWS, i) for i in range(len(users))])
    probes = emb[probe_rows] if r is None else emb[probe_rows] @ r
    return float(ax.rank1_per_user(gallery, probes, probe_user, len(users)).mean())


def _fit(emb, rows, fit_users, m, permute=None):
    c_a = ax.centroids(emb, [rows[(u, "A")] for u in fit_users])
    c_b = ax.centroids(emb, [rows[(u, "B")] for u in fit_users])
    if permute is not None:
        c_b = c_b[permute.permutation(len(c_b))]
    if m is None:
        return ax.procrustes(c_b, c_a)
    pooled = ax.l2(emb[np.concatenate([rows[(u, a)] for u in fit_users for a in ("A", "B")])])
    return ax.subspace_procrustes(c_b, c_a, ax.pca_basis(pooled, m))


def test_procrustes_recovers_a_known_rotation_when_determined():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, D))
    q = _random_orthogonal(rng)
    r = ax.procrustes(x @ q, x)          # maps B=xQ back onto A=x, so R must be Q^T
    assert np.allclose(r, q.T, atol=1e-8)


def test_low_rank_fixture_train_fitted_alignment_carries_to_unseen_users():
    rng = np.random.default_rng(2)
    emb, rows, q = _make_fixture(rng, k=16)
    test, fit = list(range(32, 49)), list(range(0, 32))
    unaligned = _score(emb, rows, test, None)
    assert unaligned < 0.25, f"fixture is not scrambled: unaligned rank-1 {unaligned}"   # assert on the fixture
    # A lives in a 16-d subspace S and B in S.Q, so the pooled basis needs S (+) S.Q: m=32.
    ceiling = _score(emb, rows, test, _fit(emb, rows, test, 32))
    trained = _score(emb, rows, test, _fit(emb, rows, fit, 32))
    null = _score(emb, rows, test, _fit(emb, rows, fit, 32, permute=np.random.default_rng(9)))
    assert ceiling > 0.9, ceiling
    assert trained > 0.9, trained
    assert null < unaligned + 0.15, (null, unaligned)


def test_isotropic_fixture_train_fitted_alignment_fails_where_test_fitted_succeeds():
    rng = np.random.default_rng(3)
    emb, rows, q = _make_fixture(rng, k=None)
    test, fit = list(range(32, 49)), list(range(0, 32))
    unaligned = _score(emb, rows, test, None)
    assert unaligned < 0.25, unaligned
    ceiling = _score(emb, rows, test, _fit(emb, rows, test, 17))
    trained = _score(emb, rows, test, _fit(emb, rows, fit, 32))
    assert ceiling > 0.9, ceiling
    # 32 fitting users span 32 of 128 dimensions; the test users' identity lives mostly
    # outside it, so the honest fit cannot carry. This is the registration's stated risk.
    assert trained < 0.5, trained


def test_unrestricted_procrustes_is_arbitrary_off_the_fitting_span():
    rng = np.random.default_rng(4)
    n, d = 32, D
    c_a, c_b = ax.l2(rng.standard_normal((n, d))), ax.l2(rng.standard_normal((n, d)))
    r1 = ax.procrustes(c_b, c_a)
    # A second exact solution: rotate within the complement of span(rows of C_A) after R1.
    # It leaves C_B @ R unchanged on the fitting rows and moves everything off the span.
    basis_a = np.linalg.qr(c_a.T)[0]                       # d x n, spans rows of C_A
    comp = np.linalg.qr(np.eye(d) - basis_a @ basis_a.T)[0][:, : d - n]
    small = _random_orthogonal(rng, d - n)
    twist = comp @ small @ comp.T + basis_a @ basis_a.T
    r2 = r1 @ twist
    assert np.allclose(np.linalg.norm(c_b @ r1 - c_a), np.linalg.norm(c_b @ r2 - c_a), atol=1e-8)
    assert np.allclose(c_b @ r1, c_b @ r2, atol=1e-8)          # identical on the fitting rows
    off = np.linalg.qr(np.eye(d) - np.linalg.qr(c_b.T)[0] @ np.linalg.qr(c_b.T)[0].T)[0][:, 0]
    assert np.linalg.norm(off @ r1 - off @ r2) > 0.5          # different off the span
    # And the subspace form is invariant to that arbitrariness on its complement.
    basis = ax.pca_basis(np.concatenate([c_a, c_b]), 16)
    rs = ax.subspace_procrustes(c_b, c_a, basis)
    assert np.allclose(rs.T @ rs, np.eye(d), atol=1e-8)
    v = off - basis @ (basis.T @ off)                         # a vector in the complement
    assert np.allclose(v @ rs, v, atol=1e-8)                  # identity there


def test_rank1_scorer_ties_are_rank_averaged_and_perfect_is_one():
    g = np.eye(5)
    probes = np.repeat(g, 3, axis=0)
    users = np.repeat(np.arange(5), 3)
    assert ax.rank1_per_user(g, probes, users, 5).tolist() == [1.0] * 5
    constant = np.ones((15, 5))
    assert ax.rank1_per_user(g, constant, users, 5).max() == 0.0    # a constant scorer is not rank-1


def test_bootstrap_ci_contains_the_mean_and_narrows_with_n():
    rng = np.random.default_rng(5)
    small = rng.random(17)
    mean, lo, hi = ax.bootstrap_mean(small, 2000, rng)
    assert lo <= mean <= hi
    big = np.tile(small, 10)
    _, lo2, hi2 = ax.bootstrap_mean(big, 2000, rng)
    assert (hi2 - lo2) < (hi - lo)

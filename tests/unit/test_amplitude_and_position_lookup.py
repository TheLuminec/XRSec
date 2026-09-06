"""
The two training-free baselines every run records beside the model (GENERALISATION_PROPOSAL 9.14).

`lookup_auc` is the mean-position lookup on the windows as the model sees them. Under `dyn`
every window mean is zero to rounding, so that column ranks rounding residue whose size
tracks movement amplitude - not a static baseline. Two columns fix that: the same lookup on
each window's RECORDED mean position, kept on the index from before any encoding and
standardised with the position channels, and movement amplitude alone, the dynamics
branch's own one-number baseline.
"""
import pathlib

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SiameseDataset, position_channel_slice
from eval import evaluate
from metrics import amplitude_lookup, movement_amplitude
from normalization import ChannelNormalizer
from results_log import FIELDS, summarize


pytestmark = pytest.mark.unit

FIXTURE_USERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "users"


class _ConstantModel(nn.Module):
    """Scores every pair the same; the baselines are what is under test."""

    head = "cosine"

    def forward(self, x1, x2):
        return torch.zeros(x1.shape[0], device=x1.device)


def _dataset(encoding: str) -> SiameseDataset:
    return SiameseDataset(str(FIXTURE_USERS_DIR), samples_per_user=8, sample_time=1,
                          sample_rate=10, seed=3, encoding=encoding)


def _metrics(dataset: SiameseDataset) -> dict:
    loader = DataLoader(dataset, batch_size=5, shuffle=False)
    _, _, metrics = evaluate(_ConstantModel(), loader, nn.BCEWithLogitsLoss(),
                             torch.device("cpu"), return_metrics=True)
    return metrics


# --- movement amplitude -------------------------------------------------------

def test_a_still_head_has_zero_amplitude_and_a_moving_one_does_not():
    still = torch.zeros(2, 7, 20)
    still[:, 4:7] = torch.tensor([0.3, 1.6, -0.2]).view(1, 3, 1)      # standing somewhere, not moving
    moving = still.clone()
    moving[1, 6] += torch.linspace(-0.1, 0.1, 20)                       # one window sways
    amplitude = movement_amplitude(moving, position_channel_slice(7))
    assert torch.allclose(movement_amplitude(still, position_channel_slice(7)), torch.zeros(2, dtype=torch.float64), atol=1e-7)
    assert amplitude[0] < 1e-7 < amplitude[1]


def test_amplitude_ignores_where_the_window_sits():
    """The whole point: a dynamics feature, invariant to the static cue."""
    g = torch.Generator().manual_seed(0)
    window = torch.randn(3, 7, 20, generator=g)
    shifted = window.clone()
    shifted[:, 4:7] += torch.tensor([5.0, -2.0, 30.0]).view(1, 3, 1)
    channels = position_channel_slice(7)
    assert torch.allclose(movement_amplitude(window, channels), movement_amplitude(shifted, channels), atol=1e-4)


def test_amplitude_lookup_is_a_similarity():
    a = torch.tensor([0.10, 0.10])
    b = torch.tensor([0.10, 0.40])
    scores = amplitude_lookup(a, b)
    assert scores[0] == 0.0 and scores[1] < scores[0]


# --- recorded positions on the index -------------------------------------------

def test_the_index_keeps_the_recorded_mean_position_under_dyn():
    """Under dyn the encoded window mean is ~0; the recorded mean must survive beside it."""
    index = _dataset("dyn").sample_index
    encoded_means = index.samples[:, 4:7].mean(dim=2)
    assert encoded_means.abs().max() < 1e-4, "dyn did not centre the windows"
    assert index.window_mean_positions.shape == (index.sample_count, 3)
    assert index.window_mean_positions.norm(dim=1).min() > 1e-3, "the recorded means were lost"


def test_the_index_amplitude_is_the_harness_definition_and_the_same_under_dyn():
    """9.14's table: float64 norm of the per-axis sd of position, before standardisation."""
    raw = _dataset("raw").sample_index
    dyn = _dataset("dyn").sample_index
    harness = raw.samples[:, 4:7].double().std(dim=2).norm(dim=1)
    assert raw.window_amplitudes.dtype == torch.float64
    assert torch.allclose(raw.window_amplitudes, harness)
    assert torch.allclose(dyn.window_amplitudes, harness, rtol=1e-4), "rotation into the mean pose must not change it"
    ChannelNormalizer("per_dataset").fit_transform(raw)
    assert torch.allclose(raw.window_amplitudes, harness), "standardisation must leave it alone"


def test_the_recorded_means_are_standardised_on_the_corpus_at_build_and_the_normaliser_leaves_them():
    """
    The 9.10 definition: per dataset, by the mean and sd of the recorded position frames.
    That equals what a raw index standardised by a target-fit normaliser gives - and the
    normaliser must not touch them again, or a dyn index would rescale them by the
    residual spread and move the number.
    """
    index = _dataset("raw").sample_index
    frames = index.samples[:, 4:7].double()
    expected = ((frames.mean(dim=2) - frames.mean(dim=(0, 2))) / frames.std(dim=(0, 2), unbiased=False)).float()
    assert torch.allclose(index.window_mean_positions, expected, atol=1e-4)
    before = index.window_mean_positions.clone()
    ChannelNormalizer("per_dataset").fit_transform(index)
    assert torch.equal(index.window_mean_positions, before), "the normaliser must leave them alone"
    assert torch.allclose(index.window_mean_positions, index.samples[:, 4:7].mean(dim=2), atol=1e-4)


def test_the_recorded_means_under_dyn_are_standardised_like_the_raw_ones():
    """A dyn index and a raw index of the same corpus must agree on them exactly."""
    raw = _dataset("raw").sample_index
    dyn = _dataset("dyn").sample_index
    assert torch.allclose(raw.window_mean_positions, dyn.window_mean_positions, atol=1e-6)
    ChannelNormalizer("per_dataset").fit_transform(dyn)
    assert torch.allclose(raw.window_mean_positions, dyn.window_mean_positions, atol=1e-6)


# --- evaluate() records both, on the same pairs -----------------------------------

def test_evaluate_records_both_baselines_pooled_and_per_dataset():
    metrics = _metrics(_dataset("raw"))
    for key in ("position_lookup_auc", "position_lookup_eer", "amplitude_auc", "amplitude_eer"):
        assert key in metrics and 0.0 <= metrics[key] <= 1.0
    (entry,) = metrics["by_dataset"].values()
    assert 0.0 <= entry["position_lookup_auc"] <= 1.0 and 0.0 <= entry["amplitude_auc"] <= 1.0


def test_under_raw_the_recorded_position_lookup_is_the_old_lookup():
    """The new column reproduces the old one where the old one was valid."""
    metrics = _metrics(_dataset("raw"))
    assert abs(metrics["position_lookup_auc"] - metrics["lookup_auc"]) < 1e-3


def test_the_recorded_position_lookup_does_not_depend_on_the_encoding():
    """
    Same fixture, same seed, same pairs: the static baseline must read the same whether the
    model was shown raw windows or dyn ones. That is what makes it a baseline for a dyn row.
    """
    raw = _metrics(_dataset("raw"))
    dyn = _metrics(_dataset("dyn"))
    assert abs(raw["position_lookup_auc"] - dyn["position_lookup_auc"]) < 1e-6


def test_a_shuffled_loader_gets_no_position_lookup_rather_than_a_misaligned_one():
    loader = DataLoader(_dataset("raw"), batch_size=5, shuffle=True)
    _, _, metrics = evaluate(_ConstantModel(), loader, nn.BCEWithLogitsLoss(),
                             torch.device("cpu"), return_metrics=True)
    assert "position_lookup_auc" not in metrics
    assert "amplitude_auc" not in metrics


# --- the results row -------------------------------------------------------------

def test_the_results_row_carries_both_baselines():
    for field in ("position_lookup_auc", "position_lookup_eer", "position_lookup_auc_by_dataset",
                  "amplitude_auc", "amplitude_eer", "amplitude_auc_by_dataset"):
        assert field in FIELDS
    history = {
        "selected_test_auc": 0.62, "lookup_auc": 0.52, "lookup_eer": 0.48,
        "position_lookup_auc": 0.73, "position_lookup_eer": 0.33,
        "amplitude_auc": 0.57, "amplitude_eer": 0.45,
        "selected_test_by_dataset": {
            "NJIT": {"auc": 0.533, "eer": 0.47, "lookup_auc": 0.523, "position_lookup_auc": 0.653,
                     "amplitude_auc": 0.590, "tier": 1, "pairs": 10},
        },
    }
    row = summarize("train", history)
    assert row["position_lookup_auc"] == 0.73 and row["amplitude_auc"] == 0.57
    assert row["position_lookup_auc_by_dataset"] == "NJIT=0.6530"
    assert row["amplitude_auc_by_dataset"] == "NJIT=0.5900"
    assert row["lookup_auc"] == 0.52, "the old column keeps its meaning"

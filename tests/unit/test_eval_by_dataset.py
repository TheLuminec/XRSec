"""Per-dataset metrics and tiers ride on every evaluation, on the same scores."""
import pathlib

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SiameseDataset
from eval import evaluate, format_by_dataset, split_metrics_by_dataset
from results_log import summarize


pytestmark = pytest.mark.unit

FIXTURE_USERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "users"


class _PositionModel(nn.Module):
    """Scores a pair by how close their mean positions are - deterministic, no training."""

    head = "cosine"

    def forward(self, x1, x2):
        return -(x1[:, 4:7].mean(dim=2) - x2[:, 4:7].mean(dim=2)).norm(dim=1)


def _loader(shuffle: bool):
    dataset = SiameseDataset(str(FIXTURE_USERS_DIR), samples_per_user=8, sample_time=1,
                             sample_rate=10, seed=3)
    return DataLoader(dataset, batch_size=5, shuffle=shuffle)


def test_evaluate_reports_each_dataset_with_its_tier_and_the_lookup():
    loader = _loader(shuffle=False)
    _, _, metrics = evaluate(_PositionModel(), loader, nn.BCEWithLogitsLoss(),
                             torch.device("cpu"), return_metrics=True)
    by_dataset = metrics["by_dataset"]
    assert len(by_dataset) == 1
    (name, entry), = by_dataset.items()
    assert entry["pairs"] == len(loader.dataset)
    assert 0.0 <= entry["auc"] <= 1.0 and 0.0 <= entry["lookup_auc"] <= 1.0
    # The fixture is not an audited corpus, so it must say so rather than guess a tier.
    assert entry["tier"] is None
    assert name in format_by_dataset(by_dataset)


def test_a_shuffled_loader_is_not_split_because_it_cannot_be_aligned():
    """Attributing shuffled scores to manifest rows would be silently wrong."""
    loader = _loader(shuffle=True)
    scores = torch.zeros(len(loader.dataset))
    labels = loader.dataset.manifest["labels"].view(-1)
    assert split_metrics_by_dataset(loader, scores, labels) == {}


def test_the_results_row_carries_the_split_and_the_lookup():
    history = {
        "selected_test_auc": 0.71,
        "lookup_auc": 0.73, "lookup_eer": 0.33,
        "selected_test_by_dataset": {
            "ViewGauss": {"auc": 0.938, "eer": 0.13, "lookup_auc": 0.932, "tier": 1, "pairs": 10},
            "Panonut360": {"auc": 0.504, "eer": 0.50, "lookup_auc": 0.508, "tier": 2, "pairs": 10},
        },
        "unseen_datasets": {"who_is_alyx": "target_fit"},
    }
    row = summarize("train", history)
    assert row["lookup_auc"] == 0.73
    assert row["test_auc_by_dataset"] == "Panonut360=0.5040;ViewGauss=0.9380"
    assert row["lookup_auc_by_dataset"] == "Panonut360=0.5080;ViewGauss=0.9320"
    assert row["eval_tiers"] == "1,2"
    assert row["unseen_datasets"] == "who_is_alyx=target_fit"

import pytest
import torch

from normalization import ChannelNormalizer


pytestmark = pytest.mark.unit


class FakeIndex:
    """Minimal stand-in for SampleIndex: samples plus dataset provenance."""

    def __init__(self, samples, dataset_ids, dataset_names):
        self.samples = samples
        self.window_dataset_ids = torch.tensor(dataset_ids, dtype=torch.long)
        self.dataset_names = list(dataset_names)
        self.sample_count = int(samples.shape[0])


def _two_datasets():
    """Dataset A near +5 with a wide spread, dataset B near -20 with a narrow one."""
    torch.manual_seed(0)
    a = torch.randn(40, 7, 12) * 2.0 + 5.0
    b = torch.randn(60, 7, 12) * 0.05 - 20.0
    return FakeIndex(torch.cat([a, b]), [0] * 40 + [1] * 60, ["A", "B"])


def test_per_dataset_removes_the_cross_dataset_offset():
    """The whole point: after fitting, datasets are no longer separable by offset."""
    index = _two_datasets()
    before_a = index.samples[:40].mean().item()
    before_b = index.samples[40:].mean().item()
    assert abs(before_a - before_b) > 20

    ChannelNormalizer("per_dataset").fit_transform(index)

    assert index.samples[:40].mean().abs().item() < 0.1
    assert index.samples[40:].mean().abs().item() < 0.1
    assert index.samples[:40].std().item() == pytest.approx(1.0, abs=0.1)
    assert index.samples[40:].std().item() == pytest.approx(1.0, abs=0.1)


def test_per_dataset_preserves_within_dataset_user_differences():
    """Identity signal is relative position within a dataset; it must survive."""
    torch.manual_seed(0)
    tall = torch.randn(30, 7, 12) * 0.1 + 1.8      # one user, consistently higher
    short = torch.randn(30, 7, 12) * 0.1 + 1.2     # another, consistently lower
    index = FakeIndex(torch.cat([tall, short]), [0] * 60, ["A"])

    gap_before = (tall.mean() - short.mean()).abs().item()
    ChannelNormalizer("per_dataset").fit_transform(index)
    gap_after = (index.samples[:30].mean() - index.samples[30:].mean()).abs().item()

    assert gap_before > 0.5
    assert gap_after > 1.0, "standardisation must not collapse between-user separation"


def test_global_mode_keeps_datasets_apart():
    """Contrast with per_dataset: one shared transform cannot remove the offset."""
    index = _two_datasets()
    ChannelNormalizer("global").fit_transform(index)
    assert abs(index.samples[:40].mean().item() - index.samples[40:].mean().item()) > 1.0


def test_none_mode_is_a_no_op():
    index = _two_datasets()
    original = index.samples.clone()
    ChannelNormalizer("none").fit_transform(index)
    assert torch.equal(index.samples, original)


def test_evaluation_uses_training_statistics_not_its_own():
    """
    The leakage guard: a held-out split standardised with its own statistics would be
    centred on itself, erasing exactly the offset that identifies its users.
    """
    torch.manual_seed(0)
    train = FakeIndex(torch.randn(50, 7, 12) * 2.0 + 5.0, [0] * 50, ["A"])
    held_out = FakeIndex(torch.randn(20, 7, 12) * 2.0 + 8.0, [0] * 20, ["A"])

    normalizer = ChannelNormalizer("per_dataset").fit(train)
    normalizer.transform(train)
    normalizer.transform(held_out)

    # Held-out data sat 3 units above the training mean, so it must stay above zero.
    assert held_out.samples.mean().item() > 0.5
    assert train.samples.mean().abs().item() < 0.1


def test_unknown_dataset_falls_back_and_warns(capsys):
    index = _two_datasets()
    normalizer = ChannelNormalizer("per_dataset").fit(index)

    other = FakeIndex(torch.randn(10, 7, 12) + 3.0, [0] * 10, ["Unseen"])
    normalizer.transform(other)

    assert "no training statistics for dataset 'Unseen'" in capsys.readouterr().out
    assert other.samples.mean().abs().item() < 0.2


def test_state_round_trip_reproduces_the_transform():
    index = _two_datasets()
    normalizer = ChannelNormalizer("per_dataset").fit(index)

    restored = ChannelNormalizer.from_state(normalizer.state_dict())
    assert restored.mode == "per_dataset"

    a, b = FakeIndex(index.samples.clone(), index.window_dataset_ids.tolist(), ["A", "B"]), \
           FakeIndex(index.samples.clone(), index.window_dataset_ids.tolist(), ["A", "B"])
    normalizer.transform(a)
    restored.transform(b)
    assert torch.allclose(a.samples, b.samples, atol=1e-6)


def test_missing_state_means_no_normalization():
    assert ChannelNormalizer.from_state(None).mode == "none"
    assert not ChannelNormalizer.from_state(None).enabled


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="normalize must be one of"):
        ChannelNormalizer("sideways")


def test_constant_channel_does_not_divide_by_zero():
    index = FakeIndex(torch.ones(20, 7, 12) * 3.0, [0] * 20, ["A"])
    ChannelNormalizer("per_dataset").fit_transform(index)
    assert torch.isfinite(index.samples).all()


def test_empty_index_is_handled():
    index = FakeIndex(torch.empty(0, 7, 12), [], ["A"])
    normalizer = ChannelNormalizer("per_dataset").fit_transform(index)
    assert normalizer.statistics == {}

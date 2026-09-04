import types

import pytest
import torch

from identity_train import AMSoftmaxHead, WindowDataset, calibrate_cosine_head
from model import SiameseModel, create_model


pytestmark = pytest.mark.unit


def _index(users=3, per_user=5, seq_len=10):
    samples = torch.randn(users * per_user, 7, seq_len)
    user_sample_indices = [
        torch.arange(u * per_user, (u + 1) * per_user, dtype=torch.long) for u in range(users)
    ]
    return types.SimpleNamespace(
        samples=samples,
        sample_count=samples.shape[0],
        num_users=users,
        user_sample_indices=user_sample_indices,
    )


# --- window dataset ----------------------------------------------------------

def test_window_dataset_labels_each_window_with_its_user():
    dataset = WindowDataset(_index(users=3, per_user=4))

    assert len(dataset) == 12
    assert dataset.num_classes == 3
    assert [int(dataset[i][1]) for i in range(12)] == [0] * 4 + [1] * 4 + [2] * 4


def test_window_dataset_uses_every_window_not_every_pair():
    """The point of the objective: one example per window, not per pair."""
    index = _index(users=5, per_user=20)
    assert len(WindowDataset(index)) == index.sample_count == 100


# --- AM-Softmax --------------------------------------------------------------

def test_amsoftmax_subtracts_the_margin_from_the_true_class_only():
    head = AMSoftmaxHead(embedding_dim=8, num_classes=4, margin=0.35, scale=1.0)
    embeddings = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    with torch.no_grad():
        plain = head(embeddings)                 # no labels -> no margin
        with_margin = head(embeddings, labels)

    for row, label in enumerate(labels.tolist()):
        assert with_margin[row, label].item() == pytest.approx(plain[row, label].item() - 0.35, abs=1e-5)
        for other in range(4):
            if other != label:
                assert with_margin[row, other].item() == pytest.approx(plain[row, other].item(), abs=1e-5)


def test_amsoftmax_output_is_bounded_by_the_scale():
    """Cosine logits live in [-s, s]; unscaled they cannot drive cross-entropy."""
    head = AMSoftmaxHead(embedding_dim=8, num_classes=4, margin=0.0, scale=30.0)
    logits = head(torch.randn(10, 8))
    assert logits.abs().max() <= 30.0 + 1e-4


def test_amsoftmax_is_invariant_to_embedding_magnitude():
    """Both embeddings and class weights are normalised, so only direction matters."""
    torch.manual_seed(0)
    head = AMSoftmaxHead(embedding_dim=8, num_classes=4, margin=0.35, scale=30.0)
    embeddings = torch.randn(5, 8)
    assert torch.allclose(head(embeddings), head(embeddings * 17.0), atol=1e-4)


def test_amsoftmax_can_learn_to_separate_identities():
    torch.manual_seed(0)
    head = AMSoftmaxHead(embedding_dim=4, num_classes=3, margin=0.2, scale=10.0)
    embeddings = torch.randn(3, 4) * 3
    labels = torch.tensor([0, 1, 2])

    optimizer = torch.optim.Adam(head.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    first = criterion(head(embeddings, labels), labels).item()
    for _ in range(200):
        optimizer.zero_grad()
        criterion(head(embeddings, labels), labels).backward()
        optimizer.step()
    assert criterion(head(embeddings, labels), labels).item() < first


# --- cosine head -------------------------------------------------------------

def test_cosine_head_scores_identical_inputs_at_maximum_similarity():
    model, _, _ = create_model(embedding_dim=8, seq_len=10, device=torch.device("cpu"),
                               extractor="bilstm", head="cosine")
    model.eval()
    x = torch.randn(4, 7, 10)
    with torch.no_grad():
        same = model(x, x)
        expected = model.cosine_scale * (1.0 - model.cosine_bias)
    assert torch.allclose(same, expected.expand_as(same), atol=1e-4)


def test_cosine_head_ignores_embedding_magnitude():
    """A metric head must not be able to encode identity in vector length."""
    model = SiameseModel(torch.nn.Identity(), embedding_dim=6, head="cosine")
    e1, e2 = torch.randn(5, 6), torch.randn(5, 6)
    assert torch.allclose(model.score(e1, e2), model.score(e1 * 9.0, e2 * 0.2), atol=1e-5)


def test_diff_linear_head_keeps_its_original_parameter_names():
    """Checkpoints written before heads were selectable must still load."""
    model = SiameseModel(torch.nn.Identity(), embedding_dim=6)
    assert model.head == "diff_linear"
    assert "classifier.weight" in model.state_dict()


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError, match="head must be"):
        SiameseModel(torch.nn.Identity(), embedding_dim=6, head="magic")


# --- calibration -------------------------------------------------------------

def test_calibration_fits_the_threshold_on_training_pairs():
    """
    Cosine ranks well but says nothing about where the accept threshold belongs.
    Calibration must move the head so accuracy at logit>0 is meaningful.
    """
    torch.manual_seed(0)
    index = _index(users=4, per_user=8, seq_len=10)
    # Make each user's windows genuinely distinct so cosine carries signal.
    for user, indices in enumerate(index.user_sample_indices):
        index.samples[indices] += (user + 1) * 5.0

    model, _, _ = create_model(embedding_dim=8, seq_len=10, device=torch.device("cpu"),
                               extractor="bilstm", head="cosine")
    with torch.no_grad():
        model.cosine_scale.fill_(1.0)
        model.cosine_bias.fill_(-50.0)      # absurd threshold: everything is "same"

    from dataset import generate_pair_manifest
    manifest = generate_pair_manifest(index, pairs_per_user=16, match_ratio=0.5, seed=3)

    before = model.cosine_bias.item()
    calibrate_cosine_head(model, index, manifest, torch.device("cpu"), batch_size=16, steps=100)

    assert model.cosine_bias.item() != before
    assert abs(model.cosine_bias.item()) < abs(before), "bias should move toward the data"


def test_calibration_is_a_noop_for_the_diff_linear_head():
    index = _index()
    model = SiameseModel(torch.nn.Identity(), embedding_dim=6)
    from dataset import generate_pair_manifest
    manifest = generate_pair_manifest(index, pairs_per_user=4, match_ratio=0.5, seed=1)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    calibrate_cosine_head(model, index, manifest, torch.device("cpu"), batch_size=8)
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)


# --- checkpoint round trip ---------------------------------------------------

def test_cosine_head_survives_a_checkpoint_round_trip(tmp_path):
    from utils import load_checkpoint, save_checkpoint

    device = torch.device("cpu")
    model, _, optimizer = create_model(embedding_dim=8, seq_len=10, device=device,
                                       extractor="bilstm", head="cosine")
    path = tmp_path / "cosine.pth"
    save_checkpoint(path, model, optimizer, epoch=1)

    restored = load_checkpoint(str(path), device, seq_len=10)
    assert restored.head == "cosine"

    x1, x2 = torch.randn(3, 7, 10), torch.randn(3, 7, 10)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(x1, x2), restored(x1, x2), atol=1e-6)


def test_identity_objective_forces_the_cosine_head():
    from train import _resolve_head

    args = types.SimpleNamespace(objective="identity_softmax", head="diff_linear")
    assert _resolve_head(args) == "cosine"

    args = types.SimpleNamespace(objective="pair_bce", head="diff_linear")
    assert _resolve_head(args) == "diff_linear"


# --- balanced identity sampling -----------------------------------------------
#
# Window counts per user span 77x on the pooled corpus (34 to 2639), leaving an
# effective identity count of 193 against 312 real ones. Identity count is the axis
# measured to bind, so that is ~38% of our diversity lost to sampling.

def _lopsided_index(counts=(100, 10, 1)):
    import types
    total = sum(counts)
    offsets, start = [], 0
    for count in counts:
        offsets.append(torch.arange(start, start + count))
        start += count
    return types.SimpleNamespace(
        samples=torch.randn(total, 7, 10),
        sample_count=total,
        num_users=len(counts),
        user_sample_indices=offsets,
    )


def test_effective_identity_count_measures_the_imbalance():
    from identity_train import WindowDataset, effective_identity_count

    balanced = WindowDataset(_lopsided_index((50, 50, 50)))
    assert effective_identity_count(balanced.labels, 3) == pytest.approx(3.0)

    lopsided = WindowDataset(_lopsided_index((100, 10, 1)))
    # One identity dominates, so this corpus is worth well under three.
    assert effective_identity_count(lopsided.labels, 3) < 1.5


def test_balanced_sampling_evens_out_who_the_model_sees():
    from identity_train import create_window_loader

    index = _lopsided_index((100, 10, 1))
    loader = create_window_loader(index, batch_size=37, device=None, seed=3,
                                  balance_identities=True)
    seen = torch.cat([labels for _, labels in loader])
    counts = torch.bincount(seen, minlength=3).float()
    # Each identity should appear roughly a third of the time despite the 100:1 split.
    assert (counts / counts.sum()).min() > 0.20


def test_unbalanced_sampling_is_still_the_default():
    from identity_train import create_window_loader

    index = _lopsided_index((100, 10, 1))
    seen = torch.cat([labels for _, labels in
                      create_window_loader(index, batch_size=37, device=None, seed=3)])
    counts = torch.bincount(seen, minlength=3).float()
    assert counts[0] / counts.sum() > 0.85, "default must stay uniform over windows"
    assert int(seen.numel()) == 111, "an epoch must still cover every window exactly once"


def test_balanced_sampling_keeps_the_epoch_the_same_size():
    from identity_train import create_window_loader

    index = _lopsided_index((100, 10, 1))
    loader = create_window_loader(index, batch_size=16, device=None, seed=5,
                                  balance_identities=True)
    assert sum(int(labels.numel()) for _, labels in loader) == 111


# --- capping: equalise by trimming the surplus, not by resampling the scarce ---
#
# On the post-BOXRR corpus the 419 pre-BOXRR identities are 17.2% of 2439 and hold
# 40.8% of all windows, so uniform sampling gives them four times their share of every
# epoch's gradient. Weighted sampling fixes that WITH REPLACEMENT, which lowers the
# number of distinct windows seen; capping trims instead.

def test_capping_bounds_every_identity_without_resampling():
    from identity_train import CappedIdentitySampler, WindowDataset

    dataset = WindowDataset(_lopsided_index((100, 10, 1)))
    sampler = CappedIdentitySampler(dataset.labels, cap=10)
    drawn = list(sampler)

    assert len(drawn) == len(set(drawn)), "a window was drawn twice in one epoch"
    counts = torch.bincount(dataset.labels[torch.tensor(drawn)], minlength=3)
    assert counts.tolist() == [10, 10, 1], "each identity capped, small ones untouched"


def test_the_default_cap_is_the_median_identity_size():
    from identity_train import CappedIdentitySampler, WindowDataset

    dataset = WindowDataset(_lopsided_index((100, 30, 5)))
    assert CappedIdentitySampler(dataset.labels).cap == 30


def test_capping_makes_the_epoch_smaller_not_larger():
    """The point against weighted sampling: this gets cheaper, not more expensive."""
    from identity_train import CappedIdentitySampler, WindowDataset

    dataset = WindowDataset(_lopsided_index((100, 10, 1)))
    sampler = CappedIdentitySampler(dataset.labels, cap=10)
    assert len(sampler) == 21 < len(dataset) == 111


def test_a_fresh_subset_is_drawn_each_epoch():
    """
    The surplus is trimmed per epoch, not discarded permanently - over many epochs a
    large identity still contributes all of its windows.
    """
    from identity_train import CappedIdentitySampler, WindowDataset

    dataset = WindowDataset(_lopsided_index((100, 10, 1)))
    sampler = CappedIdentitySampler(dataset.labels, cap=10,
                                    generator=torch.Generator().manual_seed(0))
    assert set(sampler) != set(sampler)


def test_capping_raises_the_effective_identity_count():
    from identity_train import (CappedIdentitySampler, WindowDataset,
                                effective_identity_count)

    dataset = WindowDataset(_lopsided_index((100, 10, 1)))
    before = effective_identity_count(dataset.labels, 3)
    drawn = torch.tensor(list(CappedIdentitySampler(dataset.labels, cap=10)))
    after = effective_identity_count(dataset.labels[drawn], 3)
    assert after > before


@pytest.mark.parametrize("value,expected", [
    (False, "off"), (None, "off"), ("off", "off"),
    (True, "weighted"), ("weighted", "weighted"), ("cap", "cap"),
])
def test_balance_mode_accepts_the_old_booleans(value, expected):
    """Existing configs say true/false; those must keep meaning what they meant."""
    from identity_train import resolve_balance_mode

    assert resolve_balance_mode(value) == expected


def test_an_unknown_balance_mode_is_rejected():
    from identity_train import resolve_balance_mode

    with pytest.raises(ValueError, match="off, weighted or cap"):
        resolve_balance_mode("stratified")

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

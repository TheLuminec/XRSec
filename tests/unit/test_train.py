from types import SimpleNamespace

import torch

from model import create_model
from dataset import make_pair_manifest
from train import prepare_training_round, select_hard_pair_subset
from utils import save_checkpoint


def _training_args():
    return SimpleNamespace(
        seed=7,
        lr=0.01,
        embedding_dim=8,
        sample_time=1,
        sample_rate=10,
    )


def _populate_optimizer_state(model, optimizer):
    x1 = torch.randn(2, 7, 10)
    x2 = torch.randn(2, 7, 10)
    y = torch.tensor([[1.0], [0.0]])
    loss = torch.nn.BCEWithLogitsLoss()(model(x1, x2), y)
    loss.backward()
    optimizer.step()


def test_select_hard_pair_subset_prefers_high_loss_pairs():
    candidate_manifest = make_pair_manifest(
        x1_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        x2_indices=[10, 11, 12, 13, 14, 15, 16, 17],
        labels=[1, 1, 0, 0, 1, 1, 0, 0],
        anchor_user_ids=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    losses = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    subset = select_hard_pair_subset(
        candidate_manifest,
        losses,
        hard_pairs_per_user=2,
        match_ratio=0.5,
        seed=5,
    )

    assert set(subset["x1_indices"].tolist()) == {1, 3, 5, 7}
    assert subset["labels"].sum().item() == 2.0
    assert subset["anchor_user_ids"].tolist().count(0) == 2
    assert subset["anchor_user_ids"].tolist().count(1) == 2


def test_prepare_training_round_warm_starts_and_resets_optimizer(tmp_path):
    args = _training_args()
    device = torch.device("cpu")
    checkpoint_path = tmp_path / "round_000_best.pth"

    model, _, optimizer = create_model(
        embedding_dim=args.embedding_dim,
        seq_len=args.sample_time * args.sample_rate,
        lr=args.lr,
        device=device,
    )
    _populate_optimizer_state(model, optimizer)
    save_checkpoint(checkpoint_path, model, optimizer, epoch=1, extra={"checkpoint_kind": "best", "round_idx": 0})

    warm_model, _, warm_optimizer, start_epoch, history, checkpoint = prepare_training_round(
        args,
        device,
        round_idx=1,
        previous_best_path=str(checkpoint_path),
    )

    assert start_epoch == 1
    assert history["best_epoch"] == 0
    assert checkpoint["warm_start_from"] == str(checkpoint_path)
    assert len(warm_optimizer.state) == 0

    original_state = model.state_dict()
    warm_state = warm_model.state_dict()
    assert original_state.keys() == warm_state.keys()
    for key in original_state:
        assert torch.allclose(original_state[key], warm_state[key])


def test_prepare_training_round_restores_last_checkpoint_optimizer_state(tmp_path):
    args = _training_args()
    device = torch.device("cpu")
    checkpoint_path = tmp_path / "round_000_last.pth"

    model, _, optimizer = create_model(
        embedding_dim=args.embedding_dim,
        seq_len=args.sample_time * args.sample_rate,
        lr=args.lr,
        device=device,
    )
    _populate_optimizer_state(model, optimizer)
    history = {
        "train_loss": [0.4, 0.3, 0.2],
        "train_acc": [0.5, 0.6, 0.7],
        "test_loss": [0.5, 0.4, 0.3],
        "test_acc": [0.4, 0.5, 0.6],
        "best_test_acc": 0.6,
        "best_epoch": 3,
    }
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        extra={"checkpoint_kind": "last", "round_idx": 0, "history": history},
    )

    resumed_model, _, resumed_optimizer, start_epoch, resumed_history, checkpoint = prepare_training_round(
        args,
        device,
        round_idx=0,
        resume_checkpoint_path=str(checkpoint_path),
    )

    assert start_epoch == 4
    assert resumed_history["best_epoch"] == 3
    assert checkpoint["checkpoint_kind"] == "last"
    assert len(resumed_optimizer.state) > 0

    original_state = model.state_dict()
    resumed_state = resumed_model.state_dict()
    assert original_state.keys() == resumed_state.keys()
    for key in original_state:
        assert torch.allclose(original_state[key], resumed_state[key])

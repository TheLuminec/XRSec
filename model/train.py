"""
Training script for XR biometric identification model.
"""

from __future__ import annotations

import hashlib
import math
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from boost_train import resolve_paths, run_boosted_training
from dataset import create_dataloader_from_path
from eval import evaluate
from model import create_model
from utils import save_checkpoint


def _namespaceify(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespaceify(inner) for key, inner in value.items()})
    return value


def _seed_part(value) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def derive_seed(base_seed: int, *parts) -> int:
    entropy = [int(base_seed)]
    entropy.extend(_seed_part(part) for part in parts)
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_history():
    return {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "best_test_acc": 0.0,
        "best_epoch": 0,
    }


def _normalize_history(history=None):
    normalized = _default_history()
    if history:
        for key, value in history.items():
            normalized[key] = value
    normalized.setdefault("best_test_acc", 0.0)
    normalized.setdefault("best_epoch", 0)
    return normalized


def _checkpoint_extra(extra=None, **kwargs):
    payload = {}
    if extra:
        payload.update(extra)
    payload.update(kwargs)
    return payload


def _coerce_args(args):
    if isinstance(args, dict):
        args = _namespaceify(args)
    elif isinstance(args, SimpleNamespace):
        args = SimpleNamespace(**{key: _namespaceify(value) for key, value in vars(args).items()})

    if not hasattr(args, "seed"):
        args.seed = 67

    if not hasattr(args, "boosting") or args.boosting is None:
        args.boosting = SimpleNamespace(enabled=False)
    else:
        args.boosting = _namespaceify(args.boosting)

    if not hasattr(args.boosting, "enabled"):
        args.boosting.enabled = False
    if not hasattr(args, "num_workers"):
        args.num_workers = 0
    return args


def _validate_boosting_config(args):
    boosting = args.boosting
    if not boosting.enabled:
        return

    hard_fraction = float(boosting.hard_fraction)
    if hard_fraction < 0.0 or hard_fraction > 1.0:
        raise ValueError("boosting.hard_fraction must be between 0 and 1.")

    if hasattr(boosting, "refresh_fraction") and boosting.refresh_fraction is not None:
        expected_refresh = 1.0 - hard_fraction
        if not math.isclose(float(boosting.refresh_fraction), expected_refresh, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("boosting.refresh_fraction must equal 1 - boosting.hard_fraction.")

    hard_pairs_per_user = int(round(args.samples_per_user * hard_fraction))
    candidate_pairs_per_user = int(boosting.candidate_pairs_per_user)
    if candidate_pairs_per_user < hard_pairs_per_user:
        raise ValueError("boosting.candidate_pairs_per_user must be at least the hard-pair count per user.")


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x1, batch_x2 = batch_x[0].to(device), batch_x[1].to(device)
        batch_y = batch_y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        output = model(batch_x1, batch_x2)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_x1.size(0)

        predicted = (output > 0.0).float()
        correct += int((predicted == batch_y).sum().item())
        total += int(batch_y.size(0))

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(
    epochs,
    save_path,
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    device,
    start_epoch: int = 1,
    history=None,
    last_checkpoint_path=None,
    checkpoint_extra=None,
):
    """
    Run the training process.
    """
    history = _normalize_history(history)
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 64)

    best_test_acc = history["best_test_acc"]
    if history["best_epoch"] == 0 and not history["test_acc"]:
        best_test_acc = float("-inf")

    best_epoch = history["best_epoch"]

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_loss:9.4f} | {test_acc:7.2%}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            history["best_test_acc"] = best_test_acc
            history["best_epoch"] = best_epoch
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch,
                extra=_checkpoint_extra(
                    checkpoint_extra,
                    checkpoint_kind="best",
                    history=deepcopy(history),
                ),
            )

        if last_checkpoint_path is not None:
            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                epoch,
                extra=_checkpoint_extra(
                    checkpoint_extra,
                    checkpoint_kind="last",
                    history=deepcopy(history),
                ),
            )

    history["best_test_acc"] = 0.0 if best_test_acc == float("-inf") else best_test_acc
    history["best_epoch"] = best_epoch

    print(f"\nBest test accuracy: {history['best_test_acc']:.2%}")
    print(f"Model saved to: {save_path}")
    return history


def prepare_training_round(args, device, round_idx, previous_best_path=None, resume_checkpoint_path=None):
    """
    Initialize a model, optimizer, and history for a standard or boosted round.
    """
    model, criterion, optimizer = create_model(
        embedding_dim=args.embedding_dim,
        seq_len=getattr(args, "sample_time", 1) * getattr(args, "sample_rate", 10),
        lr=args.lr,
        device=device,
    )

    history = _default_history()
    start_epoch = 1
    checkpoint = {"round_idx": round_idx}

    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = _normalize_history(checkpoint.get("history"))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        checkpoint["resume_checkpoint_path"] = str(resume_checkpoint_path)
    elif previous_best_path:
        checkpoint = torch.load(previous_best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint["warm_start_from"] = str(previous_best_path)

    history = _normalize_history(history)
    return model, criterion, optimizer, start_epoch, history, checkpoint


def _run_standard_training(args, device):
    train_paths, test_paths, exclude_users = resolve_paths(args)

    print("Loading dataset...")
    train_loader, test_loader = create_dataloader_from_path(
        train_paths,
        args.batch_size,
        device,
        is_train=True,
        test_dir=test_paths if test_paths else None,
        sample_time=getattr(args, "sample_time", 1),
        sample_rate=getattr(args, "sample_rate", 10),
        samples_per_user=getattr(args, "samples_per_user", 1000),
        num_workers=getattr(args, "num_workers", 0),
        exclude_users=exclude_users,
        swap_data=getattr(args, "swap_data", False),
        test_on_excluded=getattr(args, "test_on_excluded", False),
        seed=args.seed,
    )

    model, criterion, optimizer, start_epoch, history, _ = prepare_training_round(args, device, round_idx=0)
    return run_training(
        args.epochs,
        args.save_path,
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        device,
        start_epoch=start_epoch,
        history=history,
        checkpoint_extra={"mode": "standard", "seed": int(args.seed)},
    )


def train(args):
    """
    Train the model in standard or boosted mode.
    """
    args = _coerce_args(args)
    _validate_boosting_config(args)
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if getattr(args.boosting, "enabled", False):
        return run_boosted_training(
            args,
            device,
            prepare_training_round=prepare_training_round,
            run_training=run_training,
            derive_seed=derive_seed,
            normalize_history=_normalize_history,
        )
    return _run_standard_training(args, device)

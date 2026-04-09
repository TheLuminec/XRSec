"""
Training script for XR biometric identification model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataset import (
    build_sample_index,
    concat_pair_manifests,
    create_dataloader_from_path,
    create_pair_dataloader,
    generate_pair_manifest,
    make_pair_manifest,
)
from eval import evaluate
from model import create_model
from utils import load_checkpoint, save_checkpoint


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


def _checkpoint_paths(rounds_dir: Path, round_idx: int) -> tuple[Path, Path]:
    stem = f"round_{round_idx:03d}"
    return rounds_dir / f"{stem}_best.pth", rounds_dir / f"{stem}_last.pth"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _copy_checkpoint(src: str | Path, dest: str | Path) -> None:
    src = Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


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

        predicted = (output > -1.0).float()
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


def _subset_manifest(manifest, indices):
    return {key: value[indices] for key, value in manifest.items()}


def select_hard_pair_subset(candidate_manifest, losses, hard_pairs_per_user, match_ratio, seed):
    """
    Select the hardest pairs per anchor user while preserving label balance.
    """
    if hard_pairs_per_user <= 0 or candidate_manifest["labels"].numel() == 0:
        return make_pair_manifest([], [], [], [])

    rng = np.random.default_rng(int(seed))
    labels = candidate_manifest["labels"].view(-1)
    anchor_user_ids = candidate_manifest["anchor_user_ids"].view(-1)
    losses = losses.view(-1).detach().cpu()

    positive_target = int(round(hard_pairs_per_user * float(match_ratio)))
    positive_target = min(max(positive_target, 0), hard_pairs_per_user)
    negative_target = hard_pairs_per_user - positive_target

    selected_indices = []
    unique_users = torch.unique(anchor_user_ids, sorted=True)

    def pick_top(indices, count):
        if count <= 0 or indices.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        local_losses = losses[indices].numpy()
        tie_breakers = rng.random(indices.numel())
        order = np.lexsort((tie_breakers, -local_losses))
        chosen = indices[torch.as_tensor(order[:count], dtype=torch.long)]
        return chosen

    for user_id in unique_users.tolist():
        user_indices = torch.nonzero(anchor_user_ids == user_id, as_tuple=False).view(-1)
        positive_indices = user_indices[labels[user_indices] > 0.5]
        negative_indices = user_indices[labels[user_indices] <= 0.5]

        chosen_positive = pick_top(positive_indices, min(positive_target, positive_indices.numel()))
        chosen_negative = pick_top(negative_indices, min(negative_target, negative_indices.numel()))

        chosen = torch.cat([chosen_positive, chosen_negative], dim=0)
        remaining_needed = hard_pairs_per_user - chosen.numel()
        if remaining_needed > 0:
            chosen_set = set(chosen.tolist())
            remaining_pool = torch.as_tensor(
                [idx for idx in user_indices.tolist() if idx not in chosen_set],
                dtype=torch.long,
            )
            fill = pick_top(remaining_pool, min(remaining_needed, remaining_pool.numel()))
            chosen = torch.cat([chosen, fill], dim=0)

        selected_indices.append(chosen)

    if not selected_indices:
        return make_pair_manifest([], [], [], [])

    merged_indices = torch.cat(selected_indices, dim=0)
    if merged_indices.numel() == 0:
        return make_pair_manifest([], [], [], [])

    return _subset_manifest(candidate_manifest, merged_indices)


def _score_manifest(model, sample_index, manifest, batch_size, device, num_workers, seed):
    if manifest["labels"].numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    loader = create_pair_dataloader(
        sample_index,
        manifest,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    losses = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x1, batch_x2 = batch_x[0].to(device), batch_x[1].to(device)
            batch_y = batch_y.to(device).float().view(-1)
            output = model(batch_x1, batch_x2).view(-1)
            losses.append(criterion(output, batch_y).cpu())

    return torch.cat(losses, dim=0)


def _build_round_manifest(args, device, round_idx, train_sample_index, previous_best_path=None):
    boosting = args.boosting
    match_ratio = float(boosting.match_ratio)
    samples_per_user = int(args.samples_per_user)
    hard_pairs_per_user = int(round(samples_per_user * float(boosting.hard_fraction)))
    hard_pairs_per_user = min(max(hard_pairs_per_user, 0), samples_per_user)
    refresh_pairs_per_user = samples_per_user - hard_pairs_per_user

    if round_idx == 0 or hard_pairs_per_user == 0 or previous_best_path is None:
        manifest_seed = derive_seed(args.seed, "boost", round_idx, "train_manifest")
        manifest = generate_pair_manifest(
            train_sample_index,
            pairs_per_user=samples_per_user,
            match_ratio=match_ratio,
            seed=manifest_seed,
        )
        return manifest, {
            "manifest_seed": manifest_seed,
            "hard_pairs_per_user": 0 if round_idx == 0 else hard_pairs_per_user,
            "refresh_pairs_per_user": samples_per_user if round_idx == 0 else refresh_pairs_per_user,
        }

    candidate_seed = derive_seed(args.seed, "boost", round_idx - 1, "candidate_manifest")
    refresh_seed = derive_seed(args.seed, "boost", round_idx, "refresh_manifest")
    select_seed = derive_seed(args.seed, "boost", round_idx, "hard_select")

    candidate_manifest = generate_pair_manifest(
        train_sample_index,
        pairs_per_user=int(boosting.candidate_pairs_per_user),
        match_ratio=match_ratio,
        seed=candidate_seed,
    )
    previous_model = load_checkpoint(
        previous_best_path,
        device,
        seq_len=getattr(args, "sample_time", 1) * getattr(args, "sample_rate", 10),
    )
    candidate_losses = _score_manifest(
        previous_model,
        train_sample_index,
        candidate_manifest,
        batch_size=args.batch_size,
        device=device,
        num_workers=getattr(args, "num_workers", 0),
        seed=derive_seed(args.seed, "boost", round_idx, "candidate_loader"),
    )
    hard_manifest = select_hard_pair_subset(
        candidate_manifest,
        candidate_losses,
        hard_pairs_per_user=hard_pairs_per_user,
        match_ratio=match_ratio,
        seed=select_seed,
    )
    refresh_manifest = generate_pair_manifest(
        train_sample_index,
        pairs_per_user=refresh_pairs_per_user,
        match_ratio=match_ratio,
        seed=refresh_seed,
    )
    manifest = concat_pair_manifests([hard_manifest, refresh_manifest])
    return manifest, {
        "candidate_seed": candidate_seed,
        "refresh_seed": refresh_seed,
        "select_seed": select_seed,
        "hard_pairs_per_user": hard_pairs_per_user,
        "refresh_pairs_per_user": refresh_pairs_per_user,
    }


def _resolve_paths(args):
    train_paths = getattr(args, "data_dirs", getattr(args, "data_dir", None))
    test_paths = getattr(args, "test_dirs", getattr(args, "test_dir", None))
    exclude_users = getattr(args, "exclude_users", getattr(args, "exclude_user", None))
    return train_paths, test_paths, exclude_users


def _build_train_and_validation_indexes(args):
    train_paths, test_paths, exclude_users = _resolve_paths(args)
    swap_data = getattr(args, "swap_data", False)
    test_on_excluded = getattr(args, "test_on_excluded", False)

    train_sample_index = build_sample_index(
        train_paths,
        sample_time=getattr(args, "sample_time", 1),
        sample_rate=getattr(args, "sample_rate", 10),
        exclude_users=exclude_users,
        swap_data=swap_data,
    )

    if test_paths:
        validation_sample_index = build_sample_index(
            test_paths,
            sample_time=getattr(args, "sample_time", 1),
            sample_rate=getattr(args, "sample_rate", 10),
            exclude_users=exclude_users,
            swap_data=(not swap_data if test_on_excluded else swap_data),
        )
    elif test_on_excluded:
        validation_sample_index = build_sample_index(
            train_paths,
            sample_time=getattr(args, "sample_time", 1),
            sample_rate=getattr(args, "sample_rate", 10),
            exclude_users=exclude_users,
            swap_data=not swap_data,
        )
    else:
        validation_sample_index = train_sample_index

    return train_sample_index, validation_sample_index


def _load_boost_state(state_path: Path):
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _boost_state_payload(args, round_summaries, mode, current_round=None, best_checkpoint=None):
    payload = {
        "mode": mode,
        "seed": int(args.seed),
        "best_checkpoint": best_checkpoint,
        "current_round": current_round,
        "round_summaries": round_summaries,
        "config": {
            "rounds": int(args.boosting.rounds),
            "round_epochs": int(args.boosting.round_epochs),
            "samples_per_user": int(args.samples_per_user),
            "hard_fraction": float(args.boosting.hard_fraction),
            "candidate_pairs_per_user": int(args.boosting.candidate_pairs_per_user),
            "match_ratio": float(args.boosting.match_ratio),
        },
    }
    return payload


def _run_standard_training(args, device):
    train_paths, test_paths, exclude_users = _resolve_paths(args)

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


def _run_boosted_training(args, device):
    boosting = args.boosting
    artifact_root = Path(boosting.artifact_root)
    rounds_dir = artifact_root / "rounds"
    state_path = artifact_root / "boost_state.json"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    train_sample_index, validation_sample_index = _build_train_and_validation_indexes(args)
    validation_manifest = generate_pair_manifest(
        validation_sample_index,
        pairs_per_user=int(args.samples_per_user),
        match_ratio=float(boosting.match_ratio),
        seed=derive_seed(args.seed, "boost", "validation_manifest"),
    )
    validation_loader = create_pair_dataloader(
        validation_sample_index,
        validation_manifest,
        batch_size=args.batch_size,
        device=device,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 0),
        seed=derive_seed(args.seed, "boost", "validation_loader"),
    )

    round_summaries = []
    best_checkpoint = None
    best_metric = float("-inf")
    start_round = 0
    resume_round = None
    resume_checkpoint_path = None

    if getattr(boosting, "resume", "none") != "none":
        existing_state = _load_boost_state(state_path)
        if existing_state and existing_state.get("mode") == "complete":
            return {
                "mode": "boosted",
                "best_checkpoint": existing_state.get("best_checkpoint"),
                "round_summaries": existing_state.get("round_summaries", []),
            }
        if existing_state:
            round_summaries = existing_state.get("round_summaries", [])
            if round_summaries:
                best_round = max(round_summaries, key=lambda summary: summary["best_test_acc"])
                best_checkpoint = best_round["best_checkpoint"]
                best_metric = float(best_round["best_test_acc"])
                start_round = len(round_summaries)

            resume_round = existing_state.get("current_round")
            if resume_round is not None:
                resume_round = int(resume_round)
                _, candidate_last = _checkpoint_paths(rounds_dir, resume_round)
                if candidate_last.exists():
                    resume_checkpoint_path = str(candidate_last)
                    start_round = resume_round

    for round_idx in range(start_round, int(boosting.rounds)):
        previous_best_path = None if round_idx == 0 else (
            round_summaries[round_idx - 1]["best_checkpoint"] if round_idx - 1 < len(round_summaries) else best_checkpoint
        )
        train_manifest, manifest_meta = _build_round_manifest(
            args,
            device,
            round_idx,
            train_sample_index,
            previous_best_path=previous_best_path,
        )

        train_loader = create_pair_dataloader(
            train_sample_index,
            train_manifest,
            batch_size=args.batch_size,
            device=device,
            shuffle=True,
            num_workers=getattr(args, "num_workers", 0),
            seed=derive_seed(args.seed, "boost", round_idx, "train_loader"),
        )

        round_best_path, round_last_path = _checkpoint_paths(rounds_dir, round_idx)
        _write_json(
            state_path,
            _boost_state_payload(
                args,
                round_summaries,
                mode="running",
                current_round=round_idx,
                best_checkpoint=best_checkpoint,
            ),
        )

        model, criterion, optimizer, start_epoch, history, checkpoint = prepare_training_round(
            args,
            device,
            round_idx=round_idx,
            previous_best_path=previous_best_path,
            resume_checkpoint_path=resume_checkpoint_path if resume_round == round_idx else None,
        )

        history = run_training(
            int(boosting.round_epochs),
            str(round_best_path),
            model,
            criterion,
            optimizer,
            train_loader,
            validation_loader,
            device,
            start_epoch=start_epoch,
            history=history,
            last_checkpoint_path=str(round_last_path),
            checkpoint_extra={
                "mode": "boosted",
                "round_idx": round_idx,
                "seed": int(args.seed),
                "round_manifest_meta": manifest_meta,
                "warm_start_from": checkpoint.get("warm_start_from"),
            },
        )

        round_summary = {
            "round_idx": round_idx,
            "best_epoch": int(history["best_epoch"]),
            "best_test_acc": float(history["best_test_acc"]),
            "best_checkpoint": str(round_best_path),
            "last_checkpoint": str(round_last_path),
            "train_pairs": int(train_manifest["labels"].shape[0]),
            "validation_pairs": int(validation_manifest["labels"].shape[0]),
            "manifest_meta": manifest_meta,
        }

        if round_idx < len(round_summaries):
            round_summaries[round_idx] = round_summary
        else:
            round_summaries.append(round_summary)

        if round_summary["best_test_acc"] > best_metric:
            best_metric = round_summary["best_test_acc"]
            best_checkpoint = round_summary["best_checkpoint"]

        _write_json(
            state_path,
            _boost_state_payload(
                args,
                round_summaries,
                mode="running",
                current_round=None,
                best_checkpoint=best_checkpoint,
            ),
        )
        resume_round = None
        resume_checkpoint_path = None

    if best_checkpoint is None:
        raise RuntimeError("Boosted training did not produce a checkpoint.")

    _copy_checkpoint(best_checkpoint, args.save_path)
    final_state = _boost_state_payload(
        args,
        round_summaries,
        mode="complete",
        current_round=None,
        best_checkpoint=best_checkpoint,
    )
    _write_json(state_path, final_state)
    return {
        "mode": "boosted",
        "best_checkpoint": str(best_checkpoint),
        "round_summaries": round_summaries,
    }


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
        return _run_boosted_training(args, device)
    return _run_standard_training(args, device)

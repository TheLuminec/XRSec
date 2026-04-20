"""
Boosted training helpers for XR biometric identification.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset import (
    build_sample_index,
    concat_pair_manifests,
    create_pair_dataloader,
    generate_pair_manifest,
    make_pair_manifest,
)
from utils import load_checkpoint


def resolve_paths(args):
    train_paths = getattr(args, "data_dirs", getattr(args, "data_dir", None))
    test_paths = getattr(args, "test_dirs", getattr(args, "test_dir", None))
    exclude_users = getattr(args, "exclude_users", getattr(args, "exclude_user", None))
    return train_paths, test_paths, exclude_users


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


def _build_round_manifest(args, device, round_idx, train_sample_index, previous_best_path, derive_seed):
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


def _build_train_and_validation_indexes(args):
    train_paths, test_paths, exclude_users = resolve_paths(args)
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


def _load_checkpoint_payload(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device)


def _load_round_histories(round_summaries, device, normalize_history):
    round_histories = []
    for round_summary in round_summaries:
        checkpoint_path = round_summary.get("last_checkpoint") or round_summary.get("best_checkpoint")
        checkpoint = _load_checkpoint_payload(checkpoint_path, device)
        round_histories.append(normalize_history(checkpoint.get("history")))
    return round_histories


def _boost_state_payload(args, round_summaries, mode, current_round=None, best_checkpoint=None):
    return {
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


def run_boosted_training(
    args,
    device,
    prepare_training_round,
    run_training,
    derive_seed,
    normalize_history,
):
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
    round_histories = []
    best_checkpoint = None
    best_metric = float("-inf")
    start_round = 0
    resume_round = None
    resume_checkpoint_path = None

    if getattr(boosting, "resume", "none") != "none":
        existing_state = _load_boost_state(state_path)
        if existing_state and existing_state.get("mode") == "complete":
            round_summaries = existing_state.get("round_summaries", [])
            return {
                "mode": "boosted",
                "best_checkpoint": existing_state.get("best_checkpoint"),
                "round_summaries": round_summaries,
                "round_histories": _load_round_histories(round_summaries, device, normalize_history),
            }
        if existing_state:
            round_summaries = existing_state.get("round_summaries", [])
            round_histories = _load_round_histories(round_summaries, device, normalize_history)
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
            previous_best_path,
            derive_seed,
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

        normalized_history = normalize_history(history)
        if round_idx < len(round_histories):
            round_histories[round_idx] = normalized_history
        else:
            round_histories.append(normalized_history)

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
    _write_json(
        state_path,
        _boost_state_payload(
            args,
            round_summaries,
            mode="complete",
            current_round=None,
            best_checkpoint=best_checkpoint,
        ),
    )
    return {
        "mode": "boosted",
        "best_checkpoint": str(best_checkpoint),
        "round_summaries": round_summaries,
        "round_histories": round_histories,
    }

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
from omegaconf import OmegaConf

from boost_train import resolve_paths, run_boosted_training
from dataset import count_single_session_users, create_dataloader_from_path
from eval import evaluate
from model import DEFAULT_EXTRACTOR, create_model
from user_profile import channel_count
from utils import save_checkpoint


BOOSTING_RETIRED = """boosting is retired and will not run.

It has three protocol problems that were fixed for standard training and never for
it, and one it cannot fix:

  1. Best-round selection reads the same held-out set it reports, which inflates the
     result by about +0.02 (measured: a random-output extractor scores 0.4973 at its
     final epoch but 0.5173 as a best-of-20). Standard training solves this with
     val_user_fraction; boosting has no equivalent.
  2. boosting.artifact_root is relative, so under hydra job.chdir it lands inside a
     fresh run directory and boosting.resume never finds prior state.
  3. It is pairwise-only, so objective=identity_softmax does not apply - and that
     objective is worth +6.5 points on every backbone, 5/5 folds. Any boosted number
     is therefore not comparable to a current one.

Across every recorded run it never produced a competitive result. Use standard
training:

  mode=train objective=identity_softmax val_user_fraction=0.25

The code remains in model/boost_train.py for reference. If you want it back, the
three items above are what to fix first."""


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
        "test_auc": [],
        "test_eer": [],
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
    val_loader=None,
    start_epoch: int = 1,
    history=None,
    last_checkpoint_path=None,
    checkpoint_extra=None,
    train_epoch_fn=None,
):
    """
    Run the training process.

    Args:
        train_epoch_fn: Alternative per-epoch training step, used by the identity
            objective. Defaults to the pairwise step. Everything after the step -
            evaluation, history, checkpointing, best-model selection - is shared.
        val_loader: Optional user-disjoint split used to CHOOSE the best epoch. When
            given, `selected_test_acc` is the test accuracy at the epoch validation
            picked, which is the honest number to report. Without it the best epoch is
            chosen on the test set itself, which inflates the result by about +0.02 -
            a max over ~20 noisy evaluations, measured with a random-output extractor
            that scores 0.4973 at its final epoch but 0.5173 as a best-of.
    """
    train_epoch_fn = train_epoch_fn or train_epoch
    history = _normalize_history(history)
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 64)

    best_selection_metric = history["best_test_acc"]
    if history["best_epoch"] == 0 and not history["test_acc"]:
        best_selection_metric = float("-inf")

    best_epoch = history["best_epoch"]

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_epoch_fn(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, metrics = evaluate(model, test_loader, criterion, device, return_metrics=True)

        # The selection signal is validation when we have one, and the test set
        # otherwise (the historical behaviour, kept for continuity).
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history.setdefault("val_loss", []).append(val_loss)
            history.setdefault("val_acc", []).append(val_acc)
            selection_metric = val_acc
        else:
            selection_metric = test_acc

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history.setdefault("test_auc", []).append(metrics["auc"])
        history.setdefault("test_eer", []).append(metrics["eer"])

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_loss:9.4f} | "
              f"{test_acc:7.2%} | {metrics['auc']:7.4f} | {metrics['eer']:6.2%}")

        # The max over every epoch of the reported set. This is what earlier runs
        # recorded, and it is optimistic by construction - a max over ~20 noisy
        # evaluations. Tracked unconditionally so it stays a genuine maximum rather
        # than "the best test score among epochs validation happened to like".
        history["best_test_acc"] = max(history.get("best_test_acc", 0.0), test_acc)

        if selection_metric > best_selection_metric:
            best_selection_metric = selection_metric
            best_epoch = epoch
            # The honest figure when a validation split exists: test accuracy at the
            # epoch chosen without looking at the test set.
            history["selected_test_acc"] = test_acc
            history["best_epoch"] = best_epoch
            history["best_test_auc"] = metrics["auc"]
            history["best_test_eer"] = metrics["eer"]
            # Same values, named for what they are: the metrics at the epoch chosen
            # without looking at the test set. AUC and EER are threshold-free and
            # insensitive to pair balance, which is where accuracy has failed twice.
            history["selected_test_auc"] = metrics["auc"]
            history["selected_test_eer"] = metrics["eer"]
            if val_loader is not None:
                history["best_val_acc"] = selection_metric
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

    if best_selection_metric == float("-inf"):
        history["best_test_acc"] = 0.0
    history.setdefault("best_test_acc", 0.0)
    history["best_epoch"] = best_epoch

    # Always report the max-over-epochs figure too, since every historical number in
    # results/runs.csv is of that kind and comparisons need to be like-for-like.
    print(f"\nMax test accuracy over epochs: {history['best_test_acc']:.2%}  (optimistic)")
    if val_loader is not None:
        print(f"Test accuracy at the validation-selected epoch {best_epoch}: "
              f"{history.get('selected_test_acc', 0.0):.2%}  <- report this one")
    if history.get("test_acc"):
        print(f"Final-epoch test accuracy: {history['test_acc'][-1]:.2%}")
    print(f"Model saved to: {save_path}")
    return history


def _resolve_head(args) -> str:
    """
    The scoring head, forced to cosine for the identity objective.

    Identity training shapes embeddings for angular comparison; scoring them with a
    learned linear layer over |e1 - e2| would discard that structure.
    """
    head = str(getattr(args, "head", "diff_linear") or "diff_linear")
    if str(getattr(args, "objective", "pair_bce")) == "identity_softmax" and head != "cosine":
        if head != "diff_linear":
            print(f"NOTE: objective=identity_softmax requires head=cosine; overriding head={head}.")
        head = "cosine"
    return head


def prepare_training_round(args, device, round_idx, previous_best_path=None, resume_checkpoint_path=None):
    """
    Initialize a model, optimizer, and history for a standard or boosted round.
    """
    extractor_params = getattr(args, "extractor_params", None)
    if extractor_params is not None and not isinstance(extractor_params, dict):
        # Hydra hands over a DictConfig; the registry wants a plain mapping.
        extractor_params = OmegaConf.to_container(extractor_params, resolve=True)

    model, criterion, optimizer = create_model(
        embedding_dim=args.embedding_dim,
        seq_len=getattr(args, "sample_time", 1) * getattr(args, "sample_rate", 10),
        lr=args.lr,
        device=device,
        extractor=getattr(args, "extractor", DEFAULT_EXTRACTOR),
        extractor_params=extractor_params,
        weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0),
        head=_resolve_head(args),
        num_channels=channel_count(str(getattr(args, "channels", "full") or "full")),
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
    train_loader, val_loader, test_loader, normalizer = create_dataloader_from_path(
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
        normalize=getattr(args, "normalize", "none"),
        within_dataset_negatives=getattr(args, "within_dataset_negatives", False),
        channels=str(getattr(args, "channels", "full") or "full"),
        cross_session_positives=bool(getattr(args, "cross_session_positives", False)),
        center_position=bool(getattr(args, "center_position", False)),
        encoding=str(getattr(args, "encoding", "raw") or "raw"),
        resample=str(getattr(args, "resample", "nearest") or "nearest"),
        max_users=getattr(args, "max_users", None),
        val_user_fraction=float(getattr(args, "val_user_fraction", 0.0) or 0.0),
        return_val=True,
        return_normalizer=True,
    )

    model, criterion, optimizer, start_epoch, history, _ = prepare_training_round(args, device, round_idx=0)

    objective = str(getattr(args, "objective", "pair_bce") or "pair_bce")
    train_epoch_fn = None
    if objective == "identity_softmax":
        from identity_train import build_identity_trainer

        # The identity objective trains on windows, not pairs, and brings its own
        # optimizer (it also optimises the AM-Softmax classifier, which is discarded
        # afterwards and never saved).
        source = train_loader.dataset
        source = source.dataset if hasattr(source, "dataset") else source
        train_epoch_fn, optimizer, _ = build_identity_trainer(
            model, source.sample_index, source.manifest, device, args
        )

    history = run_training(
        args.epochs,
        args.save_path,
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        device,
        val_loader=val_loader,
        start_epoch=start_epoch,
        history=history,
        train_epoch_fn=train_epoch_fn,
        checkpoint_extra={
            "mode": "standard",
            "objective": objective,
            "head": _resolve_head(args),
            "seed": int(args.seed),
            # Carried so evaluation applies the training-time transform rather than
            # re-deriving statistics from held-out data.
            "normalizer": normalizer.state_dict(),
        },
    )

    # How much of a "cross-session" result actually was: users with one session fall
    # back to same-session positives, so the qualification belongs with the number.
    if getattr(args, "cross_session_positives", False):
        test_dataset = test_loader.dataset
        test_dataset = getattr(test_dataset, "dataset", test_dataset)
        index = getattr(test_dataset, "sample_index", None)
        if index is not None:
            history["same_session_fallback_users"] = count_single_session_users(index)

    # Realized label balance of the reported set, so a drift is visible in the record
    # rather than only in a log nobody kept.
    evaluated = test_loader.dataset
    evaluated = getattr(evaluated, "dataset", evaluated)
    manifest = getattr(evaluated, "manifest", None)
    if manifest is not None and manifest["labels"].numel():
        history["eval_positive_fraction"] = float(manifest["labels"].mean())

    return history


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
        raise RuntimeError(BOOSTING_RETIRED)
        return run_boosted_training(
            args,
            device,
            prepare_training_round=prepare_training_round,
            run_training=run_training,
            derive_seed=derive_seed,
            normalize_history=_normalize_history,
        )
    return _run_standard_training(args, device)

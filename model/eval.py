"""
Test script for XR biometric identification model.

Loads a trained model checkpoint and evaluates accuracy on the dataset.
"""

import torch
import torch.nn as nn
from dataset import create_dataloader_from_path, dataset_tier, position_channel_slice
from metrics import pair_metrics, per_dataset_metrics, static_position_lookup
from normalization import ChannelNormalizer
from utils import load_checkpoint


def _pair_datasets(loader, count: int):
    """
    (dataset id per pair, dataset names) for a manifest-backed evaluation loader, or None.

    Pairs are attributed to their anchor user's dataset. Only valid when the loader
    walks the manifest in order - a shuffled or random-split loader cannot be aligned
    with its manifest, so it is reported pooled only rather than wrongly split.
    """
    dataset = getattr(loader, "dataset", None)
    manifest = getattr(dataset, "manifest", None)
    index = getattr(dataset, "sample_index", None)
    if manifest is None or index is None or hasattr(dataset, "indices"):
        return None
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and not isinstance(sampler, torch.utils.data.SequentialSampler):
        return None
    anchors = manifest["anchor_user_ids"].view(-1)
    if anchors.numel() != count:
        return None
    user_dataset_ids = list(getattr(index, "user_dataset_ids", []) or [])
    if len(user_dataset_ids) < int(index.num_users):
        return None
    ids = torch.tensor([user_dataset_ids[int(user)] for user in anchors.tolist()], dtype=torch.long)
    return ids, list(getattr(index, "dataset_names", []) or [])


def split_metrics_by_dataset(loader, scores, labels, lookup_scores=None) -> dict:
    """
    Per-dataset AUC/EER for the model and the lookup, each tagged with its semantics tier.

    Returns {} when the loader cannot be attributed. Pooling across tiers is announced,
    because a pooled figure over tier 1 and tier 2 averages near-perfect verification
    with chance and reads like neither.
    """
    attributed = _pair_datasets(loader, int(labels.numel()))
    if attributed is None:
        return {}
    ids, names = attributed
    by_dataset = per_dataset_metrics(scores, labels, ids, dataset_names=names)
    lookup = (per_dataset_metrics(lookup_scores, labels, ids, dataset_names=names)
              if lookup_scores is not None else {})
    for name, entry in by_dataset.items():
        entry["tier"] = dataset_tier(name)
        if name in lookup:
            entry["lookup_auc"] = lookup[name]["auc"]
            entry["lookup_eer"] = lookup[name]["eer"]
    tiers = sorted({entry["tier"] for entry in by_dataset.values() if entry["tier"] is not None})
    unaudited = sorted(name for name, entry in by_dataset.items() if entry["tier"] is None)
    if unaudited:
        print(f"  NOTE: {len(unaudited)} evaluation dataset(s) have no semantics tier "
              f"({', '.join(unaudited)}); run audit_frames.py on them and add them to "
              "dataset.DATASET_TIERS")
    if len(tiers) > 1:
        print(f"  NOTE: the pooled figure mixes semantics tiers {tiers} - read the "
              "per-dataset numbers, not the average")
    return by_dataset


def evaluate(model, loader, criterion, device, return_preds=False, return_metrics=False):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        return_preds: Whether to return predictions and labels
        return_metrics: Also return threshold-free verification metrics (AUC, EER).
            Accuracy alone is measured at the fixed logit>0 threshold and hides
            ranking quality, which is what actually matters for verification.
    """
    model.eval()
    # Same reason as the training loops: a per-batch .item() stalls the GPU every step.
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0

    all_preds = []
    all_labels = []
    all_scores = []
    score_chunks = []
    label_chunks = []
    lookup_chunks = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x1 = batch_x[0].to(device)
            batch_x2 = batch_x[1].to(device)
            batch_y = batch_y.to(device).float().view(-1)

            output = model(batch_x1, batch_x2).view(-1)
            loss = criterion(output, batch_y)

            total_loss += loss.detach() * batch_y.size(0)

            predicted = (output > 0.0).float()   # if output is logits
            correct += (predicted == batch_y).sum()
            total += int(batch_y.size(0))

            if return_metrics:
                score_chunks.append(output.detach().cpu())
                label_chunks.append(batch_y.detach().cpu())
                # The training-free baseline, on the exact same pairs. Computed here
                # rather than in a separate pass so it cannot drift onto a different
                # manifest than the model was scored on - which is the only way the
                # comparison stays honest.
                channels = position_channel_slice(batch_x1.shape[1])
                lookup_chunks.append(static_position_lookup(
                    batch_x1[:, channels].mean(dim=2),
                    batch_x2[:, channels].mean(dim=2)).detach().cpu())

            if return_preds:
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

    avg_loss = float(total_loss) / total
    accuracy = float(correct) / total

    metrics = {}
    if return_metrics:
        import torch as _torch
        all_labels_t = _torch.cat(label_chunks)
        all_scores_t = _torch.cat(score_chunks)
        metrics = pair_metrics(all_scores_t, all_labels_t)
        all_lookup_t = None
        if lookup_chunks:
            all_lookup_t = _torch.cat(lookup_chunks)
            lookup = pair_metrics(all_lookup_t, all_labels_t)
            metrics["lookup_auc"] = lookup["auc"]
            metrics["lookup_eer"] = lookup["eer"]
            # Loud, because a model that does not beat three numbers is the single most
            # important thing to notice about a run and the easiest to skim past.
            if lookup["auc"] >= metrics["auc"]:
                print(f"  NOTE: the training-free mean-position lookup scored "
                      f"{lookup['auc']:.4f} AUC against the model's {metrics['auc']:.4f} "
                      f"- the model is not beating it on this set")
        # Per dataset, with its semantics tier, on the same scores. A pooled number over
        # several corpora is an average of things that differ in kind.
        metrics["by_dataset"] = split_metrics_by_dataset(loader, all_scores_t, all_labels_t, all_lookup_t)

    if return_preds and return_metrics:
        return avg_loss, accuracy, all_preds, all_labels, metrics
    if return_preds:
        return avg_loss, accuracy, all_preds, all_labels
    if return_metrics:
        return avg_loss, accuracy, metrics
    return avg_loss, accuracy

def run_evaluation(model, test_loader, criterion, test_size, device):
    """
    Run the evaluation process.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for testing data
        criterion: Loss function
        test_size: Number of samples in the test dataset
        device: Device to evaluate on
    """
    loss, accuracy, preds, labels, metrics = evaluate(
        model, test_loader, criterion, device, return_preds=True, return_metrics=True)

    # ASCII only: Windows consoles default to cp1252, so box-drawing characters
    # crash the run with UnicodeEncodeError whenever stdout is piped or redirected.
    print(f"\n{'-' * 40}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.2%}  ({int(accuracy * test_size)}/{test_size} correct)")
    print(f"  Test AUC     : {metrics.get('auc', float('nan')):.4f}   EER {metrics.get('eer', float('nan')):.3f}"
          f"   lookup AUC {metrics.get('lookup_auc', float('nan')):.4f}")
    print(f"{'-' * 40}")

    return loss, accuracy, metrics


def excluded_users_under_eval_dirs(eval_dirs, exclude_users) -> list:
    """Excluded user paths that resolve under one of the evaluation directories."""
    from pathlib import Path
    import os
    roots = [str(Path(d).resolve()) for d in ([eval_dirs] if isinstance(eval_dirs, str) else list(eval_dirs or []))]
    out = []
    for user in (exclude_users or []):
        resolved = str(Path(user).resolve())
        if any(resolved == r or resolved.startswith(r + os.sep) for r in roots):
            out.append(user)
    return out


def _resolve_eval_split(args, checkpoint):
    """
    Decide which users to evaluate on, and say so out loud.

    A checkpoint written after this change records the split it was trained under, so
    evaluating it is correct by construction. One written before does not, and the old
    behaviour was to fall back to whatever the config happened to hold - silently, and
    towards the default 5-user split that CLAUDE.md documents as unusually easy. That
    produces a healthy-looking run with a flattering number and no way to notice.
    """
    split = checkpoint.get("eval_split") if isinstance(checkpoint, dict) else None
    use_recorded = bool(getattr(args, "use_checkpoint_split", True))

    if split and use_recorded:
        eval_dirs = split.get("test_dirs") or split.get("data_dirs")
        exclude_users = split.get("exclude_users") or []
        swap = bool(split.get("swap_data", False))
        on_excluded = bool(split.get("test_on_excluded", False))
        print(f"Evaluation split recovered from the checkpoint: "
              f"{len(exclude_users)} named users, test_on_excluded={on_excluded}")
        # Reproducing the evaluation a checkpoint was selected on is the honest thing
        # to do, even when that evaluation silently lost users to the training split's
        # exclude list (the config default names five VR_User_Behavior users). So it is
        # not refused here - but it is said, so a 43-user figure is never read as 48.
        dropped = excluded_users_under_eval_dirs(eval_dirs, exclude_users) if not on_excluded else []
        if dropped:
            listed = ", ".join(str(__import__("pathlib").Path(u).name) for u in dropped[:5])
            more = f" and {len(dropped) - 5} more" if len(dropped) > 5 else ""
            print(f"  NOTE: {len(dropped)} excluded user(s) lie under the evaluation corpus and are "
                  f"DROPPED from it ({listed}{more}); the reported figure is on the remaining users, "
                  f"exactly as at training time")
        for key in ("sample_time", "sample_rate", "encoding", "resample", "center_position"):
            recorded, current = split.get(key), getattr(args, key, None)
            if recorded is not None and current is not None and str(recorded) != str(current):
                print(f"  WARNING: {key} was {recorded} at training time, {current} now")
        return eval_dirs, exclude_users, swap, on_excluded

    eval_dirs = getattr(args, "test_dirs", None) or getattr(args, "data_dirs", None)
    exclude_users = getattr(args, "exclude_users", None) or []
    swap = bool(getattr(args, "swap_data", False))
    on_excluded = bool(getattr(args, "test_on_excluded", False))
    if not split:
        print("-" * 78)
        print("WARNING: this checkpoint records no evaluation split, so the one below "
              "comes from")
        print("         the CONFIG and is UNVERIFIED. If it is the default split, the "
              "number will")
        print("         be on 5 users - a split CLAUDE.md documents as unusually easy - "
              "and will")
        print("         not be the held-out users this model was actually trained "
              "against.")
        print(f"         config split: {len(exclude_users)} named users, "
              f"test_on_excluded={on_excluded}")
        print("-" * 78)
    return eval_dirs, exclude_users, swap, on_excluded


def window_curve_model(args, device=None):
    """
    Metrics as a function of windows aggregated per side, for a trained checkpoint.

    Reuses everything mode=test already does - rebuild the extractor, recover the
    training-time normalizer, honour the checkpoint's channel set - and then replaces
    the single evaluation with a curve over k. Needs no retraining: one forward pass
    over the evaluation windows serves every k.
    """
    from dataset import build_sample_index
    from templates import format_curve, window_curve

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seq_len = getattr(args, "sample_time", 1) * getattr(args, "sample_rate", 10)
    model, checkpoint = load_checkpoint(args.model_path, device, seq_len, return_checkpoint=True)
    normalizer = ChannelNormalizer.from_state(checkpoint.get('normalizer'),
                                              unseen=getattr(args, "eval_normalize", None))
    print(normalizer.describe())

    eval_dirs, exclude_users, swap, on_excluded = _resolve_eval_split(args, checkpoint)
    index = build_sample_index(
        eval_dirs,
        sample_time=getattr(args, "sample_time", 1),
        sample_rate=getattr(args, "sample_rate", 10),
        exclude_users=exclude_users,
        swap_data=(not swap if on_excluded else swap),
        channels=checkpoint.get("channels", str(getattr(args, "channels", "full") or "full")),
        center_position=bool(getattr(args, "center_position", False)),
    )
    normalizer.transform(index)

    # Hydra hands this over as a ListConfig, which is iterable but is not a list
    # instance, so test for iterability rather than for type.
    ks = getattr(args, "curve_k", None) or [1, 2, 4, 8, 16]
    if isinstance(ks, (str, int)):
        ks = [ks]
    # An entry may be a bare k or a [reference, probe] pair; the reference side is
    # worth substantially more, so the asymmetric points are the interesting ones.
    parsed = []
    for entry in ks:
        if isinstance(entry, (str, int, float)):
            parsed.append(int(entry))
        else:
            pair = [int(v) for v in entry]
            parsed.append((pair[0], pair[-1]))
    ks = parsed

    rows = window_curve(
        model, index, device, ks=ks,
        pairs_per_user=int(getattr(args, "samples_per_user", 512)),
        seed=getattr(args, "seed", 67),
        within_dataset_negatives=bool(getattr(args, "within_dataset_negatives", True)),
        batch_size=int(getattr(args, "batch_size", 512)),
    )

    # Printed before the curve, because it predicts the curve's shape: averaging can
    # only reduce the within-session component, so a flat result should be checked
    # against what this decomposition says to expect rather than argued about.
    from templates import (cmc_curve, embed_all, format_cmc, format_decomposition,
                           variance_decomposition)

    embeddings = embed_all(model, index, device, int(getattr(args, "batch_size", 512)))
    normalise = getattr(model, "head", "diff_linear") == "cosine"
    print("\n" + format_decomposition(variance_decomposition(embeddings, index, normalise)))

    # Closed-set identification, from the same embeddings. This is the metric the XR
    # biometrics literature reports, and it is not the one this project's headline is
    # in: verification is a two-class decision at chance 0.50, identification ranks N
    # enrolled users at chance 1/N. Reporting both is what makes a published rank-1
    # figure comparable to anything here.
    identification = cmc_curve(
        model, embeddings, index, device,
        gallery_k=int(getattr(args, "gallery_k", 8)),
        probe_k=int(getattr(args, "probe_k", 1)),
        probes_per_user=int(getattr(args, "probes_per_user", 16)),
        seed=getattr(args, "seed", 67),
        gallery_sizes=tuple(int(n) for n in (getattr(args, "gallery_sizes", None) or [])),
    )
    print("\n" + format_cmc(identification))

    # Single-session users match themselves within one recording, so the figure above
    # is an upper bound. Report the cross-session-only number beside it rather than
    # picking one - the gap between them is how much of rank-1 is session matching.
    strict = cmc_curve(
        model, embeddings, index, device,
        gallery_k=int(getattr(args, "gallery_k", 8)),
        probe_k=int(getattr(args, "probe_k", 1)),
        probes_per_user=int(getattr(args, "probes_per_user", 16)),
        seed=getattr(args, "seed", 67),
        gallery_sizes=tuple(int(n) for n in (getattr(args, "gallery_sizes", None) or [])),
        require_cross_session=True,
    )
    if strict.get("users", 0) >= 2 and strict["users"] != identification.get("users"):
        print("\nCross-session users only (single-session users excluded):")
        print(format_cmc(strict))

    print("\nWindows aggregated per side (k=1 is the single-window operating point):")
    print(format_curve(rows))

    # One row per k. A curve that lived only in stdout would repeat the mistake
    # results/runs.csv exists to prevent.
    import results_log

    tag = str(getattr(args, "_dataset_tag", "") or "curve")
    for row in rows:
        if row.get("pairs"):
            # A property of the checkpoint rather than of k, so it rides on every row
            # and a reader grouping by run gets one consistent value.
            row = dict(row, rank1=identification.get("rank1"),
                       gallery_users=identification.get("users"))
            results_log.append_run(args, row, dataset_tag=tag)
    return rows


def evaluate_model(args, device=None):
    """
    Evaluate the model pipeline.
    
    Args:
        args: Arguments for testing
        device: Device to evaluate on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    seq_len = getattr(args, "sample_time", 1) * getattr(args, "sample_rate", 10)
    model, checkpoint = load_checkpoint(args.model_path, device, seq_len, return_checkpoint=True)
    # The transform the model was trained under; refitting it here would let the
    # evaluation set shape its own normalisation. An unseen corpus follows the policy
    # the run names, and the run records which datasets that applied to.
    normalizer = ChannelNormalizer.from_state(checkpoint.get('normalizer'),
                                              unseen=getattr(args, "eval_normalize", None))
    print(normalizer.describe())
    
    eval_dirs = getattr(args, "test_dirs", None) or getattr(args, "data_dirs", None) or getattr(args, "data_dir", None)
    if eval_dirs is None:
        raise ValueError("No evaluation directories were provided. Set test_dirs or data_dirs.")
        
    exclude_users = getattr(args, "exclude_users", getattr(args, "exclude_user", None))
    test_loader = create_dataloader_from_path(
        eval_dirs, 
        args.batch_size, 
        device, 
        is_train=False,
        sample_time=getattr(args, "sample_time", 1),
        sample_rate=getattr(args, "sample_rate", 10),
        num_workers=getattr(args, "num_workers", 0),
        exclude_users=exclude_users,
        swap_data=getattr(args, "swap_data", False),
        test_on_excluded=getattr(args, "test_on_excluded", False),
        seed=getattr(args, "seed", 67),
        normalize=getattr(args, "normalize", "none"),
        normalizer=normalizer if normalizer.enabled else None,
        channels=checkpoint.get("channels", str(getattr(args, "channels", "full") or "full")),
        eval_normalize=str(getattr(args, "eval_normalize", "target_fit") or "target_fit"),
    )
    test_size = len(test_loader.dataset)

    criterion = nn.BCEWithLogitsLoss()

    loss, accuracy, metrics = run_evaluation(model, test_loader, criterion, test_size, device)
    metrics = dict(metrics or {})
    metrics["unseen_datasets"] = dict(normalizer.unseen_datasets)
    if metrics.get("by_dataset"):
        print(format_by_dataset(metrics["by_dataset"]))

    return loss, accuracy, metrics


def format_by_dataset(by_dataset: dict) -> str:
    """One line per evaluation dataset: tier, model AUC/EER, lookup AUC, pair count."""
    lines = [f"  {'dataset':<44} {'tier':>4} {'model AUC':>10} {'EER':>7} {'lookup AUC':>11} {'pairs':>7}"]
    for name, entry in sorted(by_dataset.items(), key=lambda kv: (kv[1].get("tier") or 9, kv[0])):
        tier = entry.get("tier")
        lookup = entry.get("lookup_auc")
        lines.append(f"  {name[:44]:<44} {('-' if tier is None else tier):>4} {entry['auc']:>10.4f} "
                     f"{entry['eer']:>7.3f} {('-' if lookup is None else f'{lookup:.4f}'):>11} "
                     f"{entry.get('pairs', 0):>7}")
    return "\n".join(lines)

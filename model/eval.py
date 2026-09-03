"""
Test script for XR biometric identification model.

Loads a trained model checkpoint and evaluates accuracy on the dataset.
"""

import torch
import torch.nn as nn
from dataset import create_dataloader_from_path
from metrics import pair_metrics
from normalization import ChannelNormalizer
from utils import load_checkpoint


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
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_scores = []
    score_chunks = []
    label_chunks = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x1 = batch_x[0].to(device)
            batch_x2 = batch_x[1].to(device)
            batch_y = batch_y.to(device).float().view(-1)

            output = model(batch_x1, batch_x2).view(-1)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_y.size(0)

            predicted = (output > 0.0).float()   # if output is logits
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            if return_metrics:
                score_chunks.append(output.detach().cpu())
                label_chunks.append(batch_y.detach().cpu())

            if return_preds:
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    metrics = {}
    if return_metrics:
        import torch as _torch
        metrics = pair_metrics(_torch.cat(score_chunks), _torch.cat(label_chunks))

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
    loss, accuracy, preds, labels = evaluate(model, test_loader, criterion, device, return_preds=True)

    # ASCII only: Windows consoles default to cp1252, so box-drawing characters
    # crash the run with UnicodeEncodeError whenever stdout is piped or redirected.
    print(f"\n{'-' * 40}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.2%}  ({int(accuracy * test_size)}/{test_size} correct)")
    print(f"{'-' * 40}")
        
    return loss, accuracy


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
    normalizer = ChannelNormalizer.from_state(checkpoint.get('normalizer'))
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
    # evaluation set shape its own normalisation.
    normalizer = ChannelNormalizer.from_state(checkpoint.get('normalizer'))
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
    )
    test_size = len(test_loader.dataset)

    criterion = nn.BCEWithLogitsLoss()
    
    loss, accuracy, metrics = run_evaluation(model, test_loader, criterion, test_size, device)

    return loss, accuracy, metrics

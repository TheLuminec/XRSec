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
    )
    test_size = len(test_loader.dataset)

    criterion = nn.BCEWithLogitsLoss()
    
    loss, accuracy, metrics = run_evaluation(model, test_loader, criterion, test_size, device)

    return loss, accuracy, metrics

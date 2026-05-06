import os
import sys
from pathlib import Path

import matplotlib
import torch

from model import Model, SiameseModel


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_checkpoint(checkpoint_path, device, seq_len=10):
    """
    Load the model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint
        device: Device to load the model on
        seq_len: Sequence length parameter (fallback for old models)
    """
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at '{checkpoint_path}'")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model using checkpoint params before touching the dataset
    embedding_dim = checkpoint.get('embedding_dim', 128)
    seq_len = checkpoint.get('seq_len', seq_len)
    feature_extractor = Model(
        embedding_dim=embedding_dim,
        seq_len=seq_len
    ).to(device)
    model = SiameseModel(feature_extractor, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def save_checkpoint(checkpoint_path, model, optimizer, epoch, extra=None):
    """
    Save the model checkpoint.

    Args:
        checkpoint_path: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
        extra: Optional metadata to persist alongside the checkpoint
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'embedding_dim': model.feature_extractor.embedding_dim,
        'seq_len': model.feature_extractor.seq_len,
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, checkpoint_path)


def plot_training_history(history, save_path="training_history.png"):
    """Graphs training and testing loss and accuracy over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_title('Training and Testing Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_title('Training and Testing Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.close()


def plot_boosted_training_history(round_histories, save_path="boosted_training_history.png"):
    """
    Save one aggregate boosted-training plot plus one plot per round.

    Args:
        round_histories: Iterable of dicts shaped like plot_training_history inputs.
        save_path: Path for the aggregate summary graph. Per-round graphs are saved
            in a sibling directory named after this file stem.
    Returns:
        Dict with generated plot paths.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    round_plot_dir = save_path.parent / f"{save_path.stem}_rounds"
    round_plot_dir.mkdir(parents=True, exist_ok=True)

    round_paths = []
    best_accs = []
    final_train_losses = []
    final_test_losses = []
    final_train_accs = []
    final_test_accs = []
    round_indices = []

    for round_idx, history in enumerate(round_histories):
        if not history or not history.get("train_loss"):
            continue

        round_path = round_plot_dir / f"round_{round_idx:03d}.png"
        plot_training_history(history, save_path=str(round_path))
        round_paths.append(str(round_path))
        round_indices.append(round_idx)
        best_accs.append(max(history["test_acc"]))
        final_train_losses.append(history["train_loss"][-1])
        final_test_losses.append(history["test_loss"][-1])
        final_train_accs.append(history["train_acc"][-1])
        final_test_accs.append(history["test_acc"][-1])

    if round_indices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(round_indices, final_train_losses, "b-o", label="Final Train Loss")
        axes[0].plot(round_indices, final_test_losses, "r-o", label="Final Val Loss")
        axes[0].set_title("Boosted Training Loss by Round")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(round_indices, final_train_accs, "b-o", label="Final Train Acc")
        axes[1].plot(round_indices, final_test_accs, "r-o", label="Final Val Acc")
        axes[1].plot(round_indices, best_accs, "g--o", label="Best Val Acc")
        axes[1].set_title("Boosted Training Accuracy by Round")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Boosted summary graph saved to {save_path}")
        plt.close()

    return {
        "summary_path": str(save_path),
        "round_paths": round_paths,
    }

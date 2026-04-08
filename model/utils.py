import sys
import torch
import os
import matplotlib.pyplot as plt
from model import Model, SiameseModel


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
    model = SiameseModel(feature_extractor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    """
    Save the model checkpoint.

    Args:
        checkpoint_path: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'embedding_dim': model.feature_extractor.embedding_dim,
        'seq_len': model.feature_extractor.seq_len
    }, checkpoint_path)


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

"""
Test script for XR biometric identification model.

Loads a trained model checkpoint and evaluates accuracy on the dataset.
"""

import sys
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import Model, SiameseModel
from dataset import SiameseDataset


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x1, batch_x2 = batch_x[0].to(device), batch_x[1].to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x1, batch_x2)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x1.size(0)
            predicted = (output > -1.0).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            # Squeeze to handle (batch, 1) shaping and convert to list
            all_preds.extend(predicted.cpu().squeeze(-1).tolist())
            all_labels.extend(batch_y.cpu().squeeze(-1).tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def load_checkpoint(checkpoint_path, device):
    """
    Load the model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        device: Device to load the model on
    """
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at '{checkpoint_path}'")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model using checkpoint params before touching the dataset
    embedding_dim = checkpoint.get('embedding_dim', 128)
    feature_extractor = Model(
        embedding_dim=embedding_dim
    ).to(device)
    model = SiameseModel(feature_extractor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, checkpoint

def create_dataloader_from_path(data_dir: str, batch_size: int, device, checkpoint: dict = None):
    """
    Create DataLoader for testing data.
    
    Args:
        data_dir: Path to testing data
        batch_size: Batch size
        device: Device to evaluate on
        checkpoint: Loaded checkpoint dictionary specifying normalization parameters
    """
    print("Loading dataset...")
    num_workers = 0  # Disabled on Windows due to TorchScript JIT locks during spawn
    pin_memory = device.type == 'cuda'

    dataset = SiameseDataset(data_dir)

    test_size = len(dataset)
    print(f"Test: {test_size} samples")
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    
    return test_loader, test_size

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
    loss, accuracy, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'─' * 40}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.2%}  ({int(accuracy * test_size)}/{test_size} correct)")
    print(f"{'─' * 40}")
        
    return loss, accuracy

def evaluate_model(args, device):
    """
    Evaluate the model pipeline.
    
    Args:
        args: Arguments for testing
        device: Device to evaluate on
    """
    print(f"Using device: {device}")

    model, checkpoint = load_checkpoint(args.model_path, device)
    
    test_loader, test_size = create_dataloader_from_path(args.data_dir, args.batch_size, device, checkpoint)

    criterion = nn.BCEWithLogitsLoss()
    
    loss, accuracy = run_evaluation(model, test_loader, criterion, test_size, device)
    
    return loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, checkpoint = load_checkpoint('saved_tests/trained_model.pth', device)
    
    test_loader, test_size = create_dataloader_from_path(
        'datasets/ViewGauss_Head-Movement_Dataset/processed_data/users',
        2048,
        device,
        checkpoint
    )

    criterion = nn.BCEWithLogitsLoss()
    
    loss, accuracy = run_evaluation(model, test_loader, criterion, test_size, device)
    
    return loss, accuracy

if __name__ == "__main__":
    main()


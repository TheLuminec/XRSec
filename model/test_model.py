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
from model import Model
from dataset import XRSecDataset


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def per_user_accuracy(preds, labels, num_users):
    user_correct = [0] * num_users
    user_total = [0] * num_users
    for p, l in zip(preds, labels):
        user_total[l] += 1
        if p == l:
            user_correct[l] += 1
    return [(user_correct[i] / user_total[i]) if user_total[i] > 0 else 0.0
            for i in range(num_users)]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained XR biometric model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to processed_data/users/ directory")
    parser.add_argument("--model-path", type=str, default="model/trained_model.pth",
                        help="Path to the saved model checkpoint (.pth)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: checkpoint not found at '{args.model_path}'")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)

    print("Loading dataset...")
    dataset = XRSecDataset(args.data_dir, canonicalize=checkpoint.get('canonicalize', True))
    if checkpoint.get('norm_mean') is not None and checkpoint.get('norm_std') is not None:
        dataset.apply_normalization(checkpoint['norm_mean'], checkpoint['norm_std'])

    test_size = len(dataset)
    print(f"Test: {test_size} samples")
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = Model(
        num_users=checkpoint['num_users'],
        seq_len=checkpoint['seq_len'],
        num_channels=checkpoint['num_channels'],
        gnn_hidden=checkpoint['gnn_hidden'],
        lstm_hidden=checkpoint['lstm_hidden'],
        gat_heads=checkpoint['gat_heads'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {args.model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    loss, accuracy, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'─' * 40}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.2%}  ({int(accuracy * test_size)}/{test_size} correct)")
    print(f"{'─' * 40}")

    per_user = per_user_accuracy(preds, labels, dataset.num_users)
    print(f"\nPer-user accuracy ({dataset.num_users} users):")
    for label_idx, acc in enumerate(per_user):
        uid = dataset.label_to_user_id[label_idx]
        print(f"  User {uid} (label {label_idx}): {acc:.2%}")


if __name__ == "__main__":
    main()

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


def per_user_accuracy(preds, labels, anchor_labels, num_users):
    user_correct = [0] * num_users
    user_total = [0] * num_users
    for p, l, anchor in zip(preds, labels, anchor_labels):
        user_total[anchor] += 1
        if p == l:
            user_correct[anchor] += 1
    return [(user_correct[i] / user_total[i]) if user_total[i] > 0 else 0.0
            for i in range(num_users)]


def evaluate_model(args, device):
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: checkpoint not found at '{args.model_path}'")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)

    print(f"Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Initialize model using checkpoint params before touching the dataset
    embedding_dim = checkpoint.get('embedding_dim', 128)
    feature_extractor = Model(
        embedding_dim=embedding_dim,
        lstm_hidden=checkpoint['lstm_hidden'],
        gnn_hidden=checkpoint['gnn_hidden'],
        gat_heads=checkpoint.get('gat_heads', 4),
    ).to(device)
    model = SiameseModel(feature_extractor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading dataset...")
    num_workers = 0  # Disabled on Windows due to TorchScript JIT locks during spawn
    pin_memory = device.type == 'cuda'

    dataset = SiameseDataset(args.data_dir)
        
    if checkpoint.get('norm_mean') is not None and checkpoint.get('norm_std') is not None:
        dataset.apply_normalization(checkpoint['norm_mean'], checkpoint['norm_std'])

    test_size = len(dataset)
    print(f"Test: {test_size} samples")
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    criterion = nn.BCEWithLogitsLoss()
    loss, accuracy, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'─' * 40}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.2%}  ({int(accuracy * test_size)}/{test_size} correct)")
    print(f"{'─' * 40}")

    anchor_labels = dataset.labels.tolist()
    per_user = per_user_accuracy(preds, labels, anchor_labels, dataset.num_users)
    print(f"\nPer-user verification accuracy ({dataset.num_users} users):")
    for label_idx, acc in enumerate(per_user):
        uid = dataset.label_to_user_id[label_idx]
        print(f"  User {uid} (label {label_idx}): {acc:.2%}")
        
    return loss, accuracy, per_user, dataset.num_users, dataset.label_to_user_id

if __name__ == "__main__":
    print("Please run this via main.py")


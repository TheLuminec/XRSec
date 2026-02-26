"""
Training script for XR biometric identification model (Section 3.1).

Usage:
    python model/train.py --epochs 20 --data-dir datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/

Case A: Random 80/20 train/test split across all data
Case B: Leave-one-video-out (TODO: future implementation)
"""

import sys
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from model import Model
from dataset import XRSecDataset


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for a single epoch over the provided data loader.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model's accuracy and loss on a validation/test set.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    """
    Main training pipeline.
    
    Initializes the dataset, splits it into training and testing sets,
    instantiates the GNN+BiLSTM model, and runs the training loop.
    Saves the best model checkpoint based on test accuracy.
    """
    parser = argparse.ArgumentParser(description="Train XR biometric model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to processed_data/users/ directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--gnn-hidden", type=int, default=32)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--save-path", type=str, default="model/trained_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = XRSecDataset(args.data_dir, leave_one_out=True, leave_out_size=1)

    # Case A: random train/test split
    # train_size = int(args.train_split * len(dataset))
    # test_size = len(dataset) - train_size
    # train_set, test_set = random_split(dataset, [train_size, test_size])
    # print(f"Train: {train_size} samples, Test: {test_size} samples")

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = Model(
        num_users=dataset.num_users,
        lstm_hidden=args.lstm_hidden,
        gnn_hidden=args.gnn_hidden,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9}")# | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 60)

    best_train_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%}")# | {test_loss:9.4f} | {test_acc:7.2%}")

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'num_users': model.num_users,
                'seq_len': model.seq_len,
                'num_channels': model.num_channels,
                'gnn_hidden': model.gnn_hidden,
                'lstm_hidden': model.lstm_hidden,
                'gat_heads': model.gat_heads,
                }, args.save_path)

    print(f"\nBest train accuracy: {best_train_acc:.2%}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()

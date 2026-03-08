"""
Training script for XR biometric identification model (Section 3.1).

Usage:
    python model/train.py --epochs 20 --data-dir datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/
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


def run_training(args, device):
    print(f"Using device: {device}")

    print("Loading dataset...")
    if args.split_method == "leave-last-out":
        train_dataset = XRSecDataset(args.data_dir, index_load=(0, -1))
        test_dataset = XRSecDataset(args.data_dir, index_load=(-1, None))
        train_size = len(train_dataset)
        test_size = len(test_dataset)
        print(f"Train (Leave-last-out): {train_size} samples, Test: {test_size} samples")
        
        norm_mean, norm_std = train_dataset.fit_normalization(range(train_size))
        test_dataset.apply_normalization(norm_mean, norm_std)
        
        train_set = train_dataset
        test_set = test_dataset
        dataset = train_dataset
    else:
        # Random split
        dataset = XRSecDataset(args.data_dir, index_load=(0, None))
        train_size = int(args.train_split * len(dataset))
        test_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)
        print(f"Train (Random): {train_size} samples, Test: {test_size} samples")
        
        norm_mean, norm_std = dataset.fit_normalization(train_set.indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = Model(
        num_users=dataset.num_users,
        lstm_hidden=args.lstm_hidden,
        gnn_hidden=args.gnn_hidden,
        gat_heads=args.gat_heads,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 64)

    best_test_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_loss:9.4f} | {test_acc:7.2%}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
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
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'seed': args.seed,
                'train_split': args.train_split,
                'split_method': args.split_method,
                }, args.save_path)

    print(f"\nBest test accuracy: {best_test_acc:.2%}")
    print(f"Model saved to: {args.save_path}")
    
    return history, dataset.num_users, dataset.label_to_user_id

if __name__ == "__main__":
    print("Please run this via main.py")


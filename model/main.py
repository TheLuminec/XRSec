"""
Main entry point for XR biometric identification project.

Provides a unified CLI for training, testing, hyperparameter tuning,
and visualization (graphing).
"""

import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from train import run_training
from test import evaluate_model

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
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="XR Biometric Model Entry Point")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode to run: 'train' or 'test'")
    
    # Data paths & splits
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to processed_data/users/ directory")
    parser.add_argument("--split-method", type=str, choices=["random", "leave-last-out"], default="random",
                        help="Data split method (random standard vs leave-last-out)")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Ratio of data for training if split-method is 'random'")
    
    # Model saving / loading
    parser.add_argument("--save-path", type=str, default="model/trained_model.pth",
                        help="Path to save the trained model (train mode)")
    parser.add_argument("--model-path", type=str, default="model/trained_model.pth",
                        help="Path to load the trained model (test mode)")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Increased default batch size to 1024 to prevent CUDA launch overhead on tiny graphs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--gnn-hidden", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Visualization & Profiling
    parser.add_argument("--graph", action="store_true",
                        help="Generate a graph of training/testing metrics (train mode only)")
    parser.add_argument("--graph-path", type=str, default="training_history.png",
                        help="Path to save the generated graph")
    parser.add_argument("--profile", action="store_true",
                        help="Run a quick 5-batch PyTorch profiler internally instead of a full epoch.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        print("=== Starting Training Mode ===")
        history, num_users = run_training(args, device)
        
        if args.graph:
            print("Generating training graph...")
            plot_training_history(history, save_path=args.graph_path)
            
    elif args.mode == "test":
        print("=== Starting Testing Mode ===")
        loss, accuracy, per_user, num_users = evaluate_model(args, device)
        # test mode already prints per-user accuracy inside evaluate_model

if __name__ == "__main__":
    main()

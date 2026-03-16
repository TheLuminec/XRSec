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

from train import train
from test import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="XR Biometric Model Entry Point")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode to run: 'train' or 'test'")
    
    # Data paths & splits
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to processed_data/users/ directory")
    
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
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Visualization & Profiling
    parser.add_argument("--graph", action="store_true",
                        help="Generate a graph of training/testing metrics (train mode only)")
    parser.add_argument("--graph-path", type=str, default="training_history.png",
                        help="Path to save the generated graph")
    parser.add_argument("--profile", action="store_true",
                        help="Run a quick 5-batch PyTorch profiler internally instead of a full epoch.")

    args = parser.parse_args()

    if args.mode == "train":
        print("=== Starting Training Mode ===")
        history = train(args)
        
        if args.graph:
            print("Generating training graph...")
            plot_training_history(history, save_path=args.graph_path)
            
    elif args.mode == "test":
        print("=== Starting Testing Mode ===")
        loss, accuracy = evaluate_model(args)
        # test mode already prints per-user accuracy inside evaluate_model

if __name__ == "__main__":
    main()

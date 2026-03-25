"""
Training script for XR biometric identification model.

"""

import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from model import Model, SiameseModel
from dataset import SiameseDataset


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

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x1, batch_x2 = batch_x[0].to(device), batch_x[1].to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x1, batch_x2)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x1.size(0)
        
        # Binary prediction accuracy threshold: exp(output) > 25% corresponds to output > -1.09
        # Assuming output > -1.0 corresponding to ~distance < 1.0
        predicted = (output > -1.0).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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
    total_loss = 0
    correct = 0
    total = 0

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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def load_checkpoint(checkpoint_path):
    """
    Load the model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model = Model(
        embedding_dim=checkpoint['embedding_dim']
    )
    siamese_model = SiameseModel(model)
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    return siamese_model

def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    """
    Save the model checkpoint.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'embedding_dim': model.feature_extractor.embedding_dim
    }, checkpoint_path)

def create_dataloader_from_path(train_path, batch_size: int, device: str, sample_time: int = 1, sample_rate: int = 10, test_path = None):
    """
    Create DataLoader for training and testing data.
    
    Args:
        train_path: Path to training data
        batch_size: Batch size
        device: Device to train on
        sample_time: Sample time for the dataset
        sample_rate: Sample rate for the dataset
        test_path: Path to testing data (if None, split train_dataset into train and test 80% train, 20% test)
    """
    train_dataset = SiameseDataset(train_path, sample_time=sample_time, sample_rate=sample_rate)
    if test_path is None:
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(
            train_dataset,
            [int(len(train_dataset) * 0.8), int(len(train_dataset) * 0.2)],
            generator=generator
        )
    else:
        test_dataset = SiameseDataset(test_path, sample_time=sample_time, sample_rate=sample_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == 'cuda')
    return train_loader, test_loader

def create_model(embedding_dim = 128, input_dim = 10, lr = 0.001, device = "cuda"):
    """
    Create the model.
    
    Args:
        embedding_dim: Dimension of the embedding space
        input_dim: Dimension of the input data
        lr: Learning rate
        device: Device to train on
    """
    feature_extractor = Model(
        embedding_dim=embedding_dim,
        seq_len=input_dim
    ).to(device)
    
    model = SiameseModel(feature_extractor).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def run_training(epochs, save_path, model, criterion, optimizer, train_loader, test_loader, device):
    """
    Run the training process.
    
    Args:
        epochs: Number of epochs to train
        save_path: Path to save the trained model
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        device: Device to train on
    """
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 64)

    best_test_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_loss:9.4f} | {test_acc:7.2%}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(save_path, model, optimizer, epoch)
            
    print(f"\nBest test accuracy: {best_test_acc:.2%}")
    print(f"Model saved to: {save_path}")
    
    return history

def train(args):
    """
    Train the model.
    
    Args:
        args: Arguments for training:
            data_dir: Path to training data
            batch_size: Batch size
            test_dir: Path to testing data (if None, split train_dataset into train and test 80% train, 20% test)
            epochs: Number of epochs to train
            save_path: Path to save the trained model
            embedding_dim: Dimension of the embedding space
            lr: Learning rate
    """
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_loader, test_loader = create_dataloader_from_path(args.data_dir, args.batch_size, device, args.test_dir)
    
    model, criterion, optimizer = create_model(args.embedding_dim, args.lr, device)
    
    history = run_training(args.epochs, args.save_path, model, criterion, optimizer, train_loader, test_loader, device)
    return history

def main():
    """
    Main function to train the model with pre-defined parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_paths = [
        "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users"
    ]

    train_loader, test_loader = create_dataloader_from_path(
        train_paths,
        2048,
        device,
        sample_time=5,
        sample_rate=10
    )
    print("Dataset loaded.")
    print("Training dataset is VR_User_Behavior_Dataset_(Spherical_Video_Streaming)")
    
    model, criterion, optimizer = create_model(128, 0.001, 50, device)
    
    run_training(20, "saved_tests/trained_model.pth", model, criterion, optimizer, train_loader, test_loader, device)

if __name__ == "__main__":
    main()


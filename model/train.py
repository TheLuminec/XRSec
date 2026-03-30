"""
Training script for XR biometric identification model.

"""
import torch
from types import SimpleNamespace

from model import create_model
from dataset import create_dataloader_from_path
from eval import evaluate
from utils import save_checkpoint

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

    train_paths = getattr(args, "data_dirs", getattr(args, "data_dir", None))
    test_paths = getattr(args, "test_dirs", getattr(args, "test_dir", None))
    exclude_paths = getattr(args, "exclude_paths", getattr(args, "exclude_path", None))

    print("Loading dataset...")
    train_loader, test_loader = create_dataloader_from_path(
        train_paths,
        args.batch_size,
        device,
        is_train=True,
        test_dir=test_paths if test_paths else None,
        sample_time=getattr(args, "sample_time", 1),
        sample_rate=getattr(args, "sample_rate", 10),
        exclude_paths=exclude_paths
    )

    model, criterion, optimizer = create_model(
        embedding_dim=args.embedding_dim,
        seq_len=getattr(args, "sample_time", 1) * getattr(args, "sample_rate", 10),
        lr=args.lr,
        device=device,
    )
    
    history = run_training(args.epochs, args.save_path, model, criterion, optimizer, train_loader, test_loader, device)
    return history

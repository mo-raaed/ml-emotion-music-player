"""
Training Script for FER2013 Emotion Recognition using ResNet-18

This script trains a ResNet-18 model from scratch on the FER2013 dataset
for 4-class emotion classification: [angry, sad, happy, neutral].

Usage:
    python -m src.training.train_fer_resnet --epochs 20 --batch-size 64
"""

import os
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import from our modules
from src.data.fer2013_dataset import create_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.models.fer_resnet import create_model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            batch_acc = 100 * correct / total
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - "
                  f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%")
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Test the model and compute metrics.
    
    Args:
        model: The neural network model
        test_loader: Test data loader
        device: Device to test on
        class_names: List of class names
        
    Returns:
        accuracy: Test accuracy
        all_labels: True labels
        all_predictions: Predicted labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100 * accuracy_score(all_labels, all_predictions)
    
    return accuracy, all_labels, all_predictions


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    train_loss: float,
    val_loss: float,
    filepath: str,
    class_names: list
):
    """
    Save model checkpoint.
    
    Args:
        model: The neural network model
        optimizer: Optimizer
        epoch: Current epoch
        val_accuracy: Validation accuracy
        train_loss: Training loss
        val_loss: Validation loss
        filepath: Path to save checkpoint
        class_names: List of class names
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'class_names': class_names,
        'num_classes': len(class_names),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(checkpoint, filepath)
    print(f"  → Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: nn.Module, optimizer: optim.Optimizer = None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        checkpoint: Dictionary containing checkpoint data
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
    
    plt.close()


def plot_training_history(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str = None
):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.close()


def main(args):
    """
    Main training function.
    """
    print("=" * 80)
    print("FER2013 Emotion Recognition Training")
    print("=" * 80)
    print(f"\nTraining Configuration:")
    print(f"  Dataset: FER2013 (4 classes)")
    print(f"  Model: ResNet-18 (Random Initialization)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  CSV Path: {args.csv_path}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 80}")
    
    # Load data
    print("\n" + "=" * 80)
    print("Loading Dataset...")
    print("=" * 80)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True
    )
    
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class names: {CLASS_NAMES}")
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating Model...")
    print("=" * 80)
    
    model, device = create_model(
        num_classes=NUM_CLASSES,
        input_channels=1,
        use_gpu_if_available=True
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    print(f"\nLoss Function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr={args.lr})")
    print(f"Scheduler: StepLR (step_size=7, gamma=0.1)")
    
    # Training tracking
    best_val_accuracy = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n{'=' * 80}")
        print(f"Epoch [{epoch}/{args.epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(args.checkpoint_dir, 'fer_resnet18_best.pth')
            save_checkpoint(
                model, optimizer, epoch, val_acc, train_loss, val_loss,
                checkpoint_path, CLASS_NAMES
            )
            print(f"  *** New best validation accuracy: {val_acc:.2f}% ***")
        
        print(f"{'=' * 80}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}% (Epoch {best_epoch})")
    
    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Load best model for testing
    print("\n" + "=" * 80)
    print("Testing Best Model...")
    print("=" * 80)
    
    best_checkpoint_path = os.path.join(args.checkpoint_dir, 'fer_resnet18_best.pth')
    checkpoint = load_checkpoint(best_checkpoint_path, model)
    
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Test on test set
    test_acc, test_labels, test_predictions = test_model(
        model, test_loader, device, CLASS_NAMES
    )
    
    print(f"\n{'=' * 80}")
    print(f"Test Set Results:")
    print(f"{'=' * 80}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Compute confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.checkpoint_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    
    # Classification report
    print(f"\n{'=' * 80}")
    print("Classification Report:")
    print(f"{'=' * 80}")
    report = classification_report(
        test_labels, test_predictions, 
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # Save classification report to file
    report_path = os.path.join(args.checkpoint_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FER2013 Emotion Recognition - Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: ResNet-18 (Random Initialization)\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("=" * 80 + "\n")
        f.write("Classification Report:\n")
        f.write("=" * 80 + "\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("=" * 80 + "\n")
        f.write(str(cm) + "\n")
    
    print(f"\nClassification report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("All Done!")
    print("=" * 80)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train ResNet-18 for FER2013 emotion recognition'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/fer2013.csv',
        help='Path to FER2013 CSV file (default: data/fer2013.csv)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models',
        help='Directory to save checkpoints (default: models)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

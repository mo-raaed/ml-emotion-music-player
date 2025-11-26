"""
Quick training script launcher with common configurations.

This script provides an easy way to start training with predefined configurations.
"""

import subprocess
import sys
import os


def main():
    """Launch training with default or custom parameters."""
    
    # Check if fer2013.csv exists
    csv_path = 'data/fer2013.csv'
    if not os.path.exists(csv_path):
        print("=" * 80)
        print("ERROR: FER2013 dataset not found!")
        print("=" * 80)
        print(f"\nPlease place the fer2013.csv file in the data/ directory.")
        print(f"Expected location: {os.path.abspath(csv_path)}")
        print("\nYou can download FER2013 from:")
        print("  - Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")
        print("=" * 80)
        sys.exit(1)
    
    print("=" * 80)
    print("FER2013 Training Launcher")
    print("=" * 80)
    print(f"\nDataset found: {csv_path}")
    
    # Default configuration
    config = {
        'epochs': 20,
        'batch_size': 64,
        'lr': 0.001,
        'num_workers': 4
    }
    
    print(f"\nDefault Configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Workers: {config['num_workers']}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # Build command
    cmd = [
        sys.executable,
        '-m',
        'src.training.train_fer_resnet',
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--num-workers', str(config['num_workers']),
        '--csv-path', csv_path,
        '--checkpoint-dir', 'models'
    ]
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)


if __name__ == '__main__':
    main()

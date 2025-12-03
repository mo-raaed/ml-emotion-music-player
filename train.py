"""
Quick training script launcher that passes through all arguments.

This script provides an easy way to start training and forwards all
command-line arguments to the actual training script.
"""

import subprocess
import sys
import os


def main():
    """Launch training and forward all command-line arguments."""
    
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
    
    # Show default configuration
    print(f"\nDefault Configuration:")
    print(f"  Epochs: 20")
    print(f"  Batch Size: 64")
    print(f"  Learning Rate: 0.001")
    print(f"  Workers: 4")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # Build command - forward all arguments from command line
    cmd = [
        sys.executable,
        '-m',
        'src.training.train_fer_resnet',
    ]
    
    # Add all command-line arguments (excluding the script name)
    cmd.extend(sys.argv[1:])
    
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

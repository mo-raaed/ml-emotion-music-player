"""
FER2013 Dataset Module for Emotion Recognition

This module handles loading and preprocessing of the FER2013 dataset,
including remapping from 7 original classes to 4 target emotion classes.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, List, Optional
from collections import Counter


# Class mapping from 7 original FER2013 classes to 4 target classes
# Original FER2013 labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
CLASS_MAPPING = {
    0: 0,  # angry -> angry
    1: 0,  # disgust -> angry
    2: 1,  # fear -> sad
    4: 1,  # sad -> sad
    3: 2,  # happy -> happy
    5: 3,  # surprise -> neutral
    6: 3,  # neutral -> neutral
}

# Target class names (4 classes, indexed 0-3)
CLASS_NAMES = ["angry", "sad", "happy", "neutral"]

# Number of classes after mapping
NUM_CLASSES = 4


class Fer2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013 emotion recognition.
    
    Args:
        dataframe: Pandas DataFrame containing the FER2013 data
        transform: Optional torchvision transforms to apply
        class_mapping: Dictionary mapping original labels to target labels
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        transform: Optional[transforms.Compose] = None,
        class_mapping: Dict[int, int] = CLASS_MAPPING
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_mapping = class_mapping
        
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Returns:
            image: Tensor of shape (1, 48, 48) with values in [0, 1]
            label: Integer label in range [0, 3]
        """
        row = self.dataframe.iloc[idx]
        
        # Parse pixel string into numpy array
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        
        # Reshape to 48x48 grayscale image
        image = pixels.reshape(48, 48)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image, mode='L')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        # Map original label to target label
        original_label = int(row['emotion'])
        target_label = self.class_mapping[original_label]
        
        return image, target_label


def load_fer2013_csv(csv_path: str) -> pd.DataFrame:
    """
    Load FER2013 CSV file.
    
    Args:
        csv_path: Path to fer2013.csv
        
    Returns:
        DataFrame with columns: emotion, pixels, Usage
    """
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    required_columns = ['emotion', 'pixels', 'Usage']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split FER2013 dataset into train, validation, and test sets based on Usage column.
    
    Args:
        df: Full FER2013 DataFrame
        
    Returns:
        train_df: Training set (Usage == "Training")
        val_df: Validation set (Usage == "PublicTest")
        test_df: Test set (Usage == "PrivateTest")
    """
    train_df = df[df['Usage'] == 'Training'].copy()
    val_df = df[df['Usage'] == 'PublicTest'].copy()
    test_df = df[df['Usage'] == 'PrivateTest'].copy()
    
    return train_df, val_df, test_df


def get_train_transforms() -> transforms.Compose:
    """
    Get training data augmentation transforms.
    Includes random horizontal flip, rotation, and normalization.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),
        transforms.ToTensor(),  # Converts to [0, 1] range and adds channel dimension
    ])


def get_val_test_transforms() -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    Only converts to tensor and normalizes.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range and adds channel dimension
    ])


def create_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        csv_path: Path to fer2013.csv
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        train_loader: Training DataLoader with augmentation
        val_loader: Validation DataLoader without augmentation
        test_loader: Test DataLoader without augmentation
    """
    # Load and split dataset
    df = load_fer2013_csv(csv_path)
    train_df, val_df, test_df = split_dataset(df)
    
    # Create datasets with appropriate transforms
    train_dataset = Fer2013Dataset(train_df, transform=get_train_transforms())
    val_dataset = Fer2013Dataset(val_df, transform=get_val_test_transforms())
    test_dataset = Fer2013Dataset(test_df, transform=get_val_test_transforms())
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataframe: pd.DataFrame, class_mapping: Dict[int, int] = CLASS_MAPPING) -> Dict[str, int]:
    """
    Get distribution of classes after mapping to target labels.
    
    Args:
        dataframe: DataFrame with 'emotion' column
        class_mapping: Mapping from original to target labels
        
    Returns:
        Dictionary mapping class names to counts
    """
    # Map original labels to target labels
    mapped_labels = dataframe['emotion'].map(class_mapping)
    
    # Count occurrences
    label_counts = Counter(mapped_labels)
    
    # Convert to class names
    distribution = {CLASS_NAMES[label]: label_counts[label] for label in sorted(label_counts.keys())}
    
    return distribution


if __name__ == "__main__":
    """
    Test block to verify dataset loading and class distribution.
    """
    import os
    
    # Path to FER2013 CSV
    csv_path = os.path.join('data', 'fer2013.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please place fer2013.csv in the data/ directory.")
        exit(1)
    
    print("=" * 60)
    print("FER2013 Dataset Loader Test")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from: {csv_path}")
    df = load_fer2013_csv(csv_path)
    print(f"Total samples loaded: {len(df)}")
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    print("\n" + "-" * 60)
    print("Dataset Split Sizes:")
    print("-" * 60)
    print(f"Training set:   {len(train_df):6,} samples")
    print(f"Validation set: {len(val_df):6,} samples")
    print(f"Test set:       {len(test_df):6,} samples")
    print(f"Total:          {len(train_df) + len(val_df) + len(test_df):6,} samples")
    
    # Original class distribution (7 classes)
    print("\n" + "-" * 60)
    print("Original FER2013 Class Distribution (7 classes):")
    print("-" * 60)
    original_classes = {
        0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
        4: "Sad", 5: "Surprise", 6: "Neutral"
    }
    for label, name in original_classes.items():
        train_count = (train_df['emotion'] == label).sum()
        val_count = (val_df['emotion'] == label).sum()
        test_count = (test_df['emotion'] == label).sum()
        total_count = train_count + val_count + test_count
        print(f"{name:10s} (label {label}): Train={train_count:5,}, Val={val_count:4,}, Test={test_count:4,}, Total={total_count:5,}")
    
    # Target class distribution (4 classes)
    print("\n" + "-" * 60)
    print("Target Class Distribution (4 classes after mapping):")
    print("-" * 60)
    print("\nClass Mapping:")
    print("  angry   <- Angry, Disgust")
    print("  sad     <- Fear, Sad")
    print("  happy   <- Happy")
    print("  neutral <- Surprise, Neutral")
    print()
    
    train_dist = get_class_distribution(train_df)
    val_dist = get_class_distribution(val_df)
    test_dist = get_class_distribution(test_df)
    
    for i, class_name in enumerate(CLASS_NAMES):
        train_count = train_dist.get(class_name, 0)
        val_count = val_dist.get(class_name, 0)
        test_count = test_dist.get(class_name, 0)
        total_count = train_count + val_count + test_count
        percentage = (total_count / len(df)) * 100
        print(f"{class_name:8s} (label {i}): Train={train_count:5,}, Val={val_count:4,}, Test={test_count:4,}, Total={total_count:5,} ({percentage:5.2f}%)")
    
    # Create dataloaders
    print("\n" + "-" * 60)
    print("Creating DataLoaders...")
    print("-" * 60)
    batch_size = 64
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path, 
        batch_size=batch_size,
        num_workers=0  # Set to 0 for testing to avoid multiprocessing issues
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")
    
    # Test loading a batch
    print("\n" + "-" * 60)
    print("Testing batch loading...")
    print("-" * 60)
    
    # Get a batch from training set
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")  # Should be (batch_size, 1, 48, 48)
    print(f"Image dtype: {images.dtype}")
    print(f"Image value range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels in batch: {labels.tolist()[:10]}...")  # Show first 10 labels
    print(f"Unique labels in batch: {sorted(labels.unique().tolist())}")
    
    print("\n" + "=" * 60)
    print("Dataset loading test completed successfully!")
    print("=" * 60)

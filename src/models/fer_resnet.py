"""
FER ResNet-18 Model for 4-class Emotion Recognition

This module defines a ResNet-18 architecture adapted for single-channel 48×48 
grayscale facial images, trained from scratch with NO pretrained weights.

Architecture:
- ResNet-18 backbone with random initialization
- Modified first conv layer for grayscale (1 channel) input
- Modified final FC layer for 4-class output: [angry, sad, happy, neutral]
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Optional


class FerResNet18(nn.Module):
    """
    ResNet-18 model adapted for FER2013 emotion recognition.
    
    This model uses the ResNet-18 architecture from torchvision with random 
    initialization only, and is NOT pretrained on any dataset. All weights 
    are learned from scratch on the FER2013 dataset.
    
    Key modifications:
    1. First convolutional layer changed from 3 input channels to 1 (grayscale)
    2. Final fully connected layer changed to output 4 classes
    
    Args:
        num_classes: Number of output classes (default: 4 for angry, sad, happy, neutral)
        input_channels: Number of input channels (default: 1 for grayscale)
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1):
        super(FerResNet18, self).__init__()
        
        # Load ResNet-18 architecture with NO pretrained weights (random initialization)
        # IMPORTANT: weights=None ensures no pretrained weights are loaded
        self.model = resnet18(weights=None)
        
        # Modify first convolutional layer for single-channel grayscale input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modified: Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Get the number of input features for the final FC layer
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer for 4-class classification
        # Original: Linear(in_features=512, out_features=1000)
        # Modified: Linear(in_features=512, out_features=4)
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
        self.input_channels = input_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 48, 48)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
            Raw logits (no softmax applied)
        """
        return self.model(x)
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters in the model.
        
        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    num_classes: int = 4, 
    input_channels: int = 1,
    use_gpu_if_available: bool = True
) -> tuple[FerResNet18, torch.device]:
    """
    Create a FER ResNet-18 model with random initialization (NO pretrained weights).
    
    This function creates a ResNet-18 model that is randomly initialized and 
    NOT pretrained on any dataset (e.g., ImageNet). All training will be done 
    from scratch on the FER2013 dataset.
    
    Args:
        num_classes: Number of emotion classes (default: 4)
        input_channels: Number of input channels (default: 1 for grayscale)
        use_gpu_if_available: Move model to GPU if available (default: True)
        
    Returns:
        model: FerResNet18 model instance
        device: torch.device where the model is located
    """
    # Determine device
    if use_gpu_if_available and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create model with random initialization
    model = FerResNet18(num_classes=num_classes, input_channels=input_channels)
    
    # Move model to device
    model = model.to(device)
    
    # Print model info
    num_params = model.get_num_parameters()
    print(f"\nModel: ResNet-18 (Random Initialization - NOT Pretrained)")
    print(f"Input: ({input_channels}, 48, 48) grayscale images")
    print(f"Output: {num_classes} classes")
    print(f"Total trainable parameters: {num_params:,}")
    
    return model, device


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'total': total,
        'frozen': total - trainable
    }


if __name__ == "__main__":
    """
    Test block to verify model architecture and forward pass.
    """
    print("=" * 70)
    print("FER ResNet-18 Model Test")
    print("=" * 70)
    
    # Create model
    print("\nCreating model...")
    model, device = create_model(
        num_classes=4, 
        input_channels=1,
        use_gpu_if_available=True
    )
    
    # Print parameter counts by layer type
    print("\n" + "-" * 70)
    print("Parameter Breakdown:")
    print("-" * 70)
    param_info = count_parameters(model)
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Frozen parameters:    {param_info['frozen']:,}")
    print(f"Total parameters:     {param_info['total']:,}")
    
    # Create dummy input batch
    print("\n" + "-" * 70)
    print("Testing forward pass...")
    print("-" * 70)
    batch_size = 2
    channels = 1
    height = 48
    width = 48
    
    # Create random input tensor
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    print(f"Input shape:  {dummy_input.shape} (batch_size, channels, height, width)")
    print(f"Input dtype:  {dummy_input.dtype}")
    print(f"Input device: {dummy_input.device}")
    
    # Set model to evaluation mode for testing
    model.eval()
    
    # Forward pass (no gradient computation needed for testing)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape} (batch_size, num_classes)")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    print(f"\nSample output (raw logits):")
    print(output)
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"\nSample output (probabilities after softmax):")
    print(probabilities)
    print(f"\nProbabilities sum to 1.0: {torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size).to(device))}")
    
    # Verify output shape
    expected_shape = (batch_size, 4)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("\n" + "=" * 70)
    print("✓ Model test completed successfully!")
    print("=" * 70)
    
    # Additional architecture details
    print("\nModel Architecture Summary:")
    print("-" * 70)
    print(f"First layer: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)")
    print(f"Backbone: ResNet-18 (4 residual blocks)")
    print(f"Final layer: Linear(512, 4)")
    print(f"Activation: ReLU")
    print(f"Normalization: BatchNorm2d")
    print(f"Pooling: MaxPool2d + AdaptiveAvgPool2d")
    print(f"\nNote: This model is randomly initialized with NO pretrained weights.")
    print("      All parameters will be learned from scratch during training.")

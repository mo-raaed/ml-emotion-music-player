"""
Model architectures module.
This module contains ResNet-18 based emotion recognition model.
"""

from .fer_resnet import (
    FerResNet18,
    create_model,
    count_parameters
)

__all__ = [
    'FerResNet18',
    'create_model',
    'count_parameters',
]

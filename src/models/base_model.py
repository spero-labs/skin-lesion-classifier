"""Base model module providing abstract interface for all architectures.

This module defines the abstract base class that all model architectures
must inherit from. It provides a consistent interface for model operations
including forward passes, feature extraction, parameter management, and
differential learning rates.

The base model ensures:
    - Consistent API across all architectures
    - Proper parameter group management for optimizers
    - Flexible backbone freezing/unfreezing for transfer learning
    - Feature extraction capabilities for interpretability

Typical usage:
    class CustomModel(BaseModel):
        def __init__(self, num_classes, dropout):
            super().__init__(num_classes, dropout)
            # Initialize architecture
        
        def forward(self, x):
            # Implement forward pass
            return logits
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all model architectures.
    
    Provides a unified interface for all models in the skin lesion
    classification system. This ensures consistent behavior across
    different architectures and simplifies model management.
    
    All derived models must implement:
        - forward(): The forward pass computation
        - (optional) get_features(): Feature extraction before classification
    
    The base class provides:
        - Parameter group management for differential learning rates
        - Backbone freezing/unfreezing for transfer learning
        - Consistent initialization interface
    
    Attributes:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for regularization
        backbone: Optional pretrained backbone network
        classifier: Classification head
    """
    
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer."""
        raise NotImplementedError("Feature extraction not implemented for this model")
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def get_param_groups(self, lr: float, lr_backbone: Optional[float] = None) -> list:
        """
        Get parameter groups for differential learning rates.
        
        Args:
            lr: Learning rate for head
            lr_backbone: Learning rate for backbone (if None, uses lr/10)
        
        Returns:
            List of parameter groups
        """
        if lr_backbone is None:
            lr_backbone = lr / 10
        
        if hasattr(self, "backbone"):
            return [
                {"params": self.backbone.parameters(), "lr": lr_backbone},
                {"params": self.classifier.parameters(), "lr": lr},
            ]
        else:
            return [{"params": self.parameters(), "lr": lr}]
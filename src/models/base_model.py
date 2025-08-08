import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
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
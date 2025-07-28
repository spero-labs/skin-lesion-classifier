import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention maps for interpretability"""
        pass

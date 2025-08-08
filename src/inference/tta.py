import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Callable
import albumentations as A


class TestTimeAugmentation:
    """Test-time augmentation for improved predictions."""
    
    def __init__(self, n_augmentations: int = 5):
        """
        Initialize TTA.
        
        Args:
            n_augmentations: Number of augmented versions to average
        """
        self.n_augmentations = n_augmentations
        self.augmentations = self._create_augmentations()
    
    def _create_augmentations(self) -> List[Callable]:
        """Create list of augmentation transforms."""
        augmentations = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            lambda x: torch.flip(x, dims=[2]),  # Vertical flip
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotate 90
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # Rotate 270
        ]
        
        return augmentations[:self.n_augmentations]
    
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA to get predictions.
        
        Args:
            model: Model for prediction
            x: Input tensor
        
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for aug_fn in self.augmentations:
            # Apply augmentation
            x_aug = aug_fn(x)
            
            # Get prediction
            with torch.no_grad():
                pred = model(x_aug)
            
            # Reverse augmentation if needed (for spatial consistency)
            # This is simplified - full implementation would reverse each aug
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
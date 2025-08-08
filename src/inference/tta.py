"""Test-time augmentation module for improving prediction robustness.

This module implements test-time augmentation (TTA), a technique that
applies multiple augmentations to test images and averages the predictions
to reduce variance and improve accuracy. TTA is particularly effective for
medical images where orientation and small variations can affect predictions.

Key concepts:
    - Multiple augmented versions of the same image
    - Predictions averaged across augmentations
    - Reduces prediction variance
    - Improves model calibration
    - No additional training required

Benefits for skin lesion classification:
    1. Reduces sensitivity to image orientation
    2. Improves robustness to minor variations
    3. Better uncertainty estimation
    4. Higher accuracy without model changes
    5. Particularly effective for borderline cases

Typical usage:
    tta = TestTimeAugmentation(n_augmentations=5)
    
    # Apply TTA during inference
    predictions = tta(model, image_tensor)
    
    # Results are averaged across augmentations
    final_prediction = predictions.argmax()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Callable
import albumentations as A


class TestTimeAugmentation:
    """Test-time augmentation for robust skin lesion predictions.
    
    Applies multiple augmentations to test images and aggregates
    predictions to reduce variance and improve accuracy. This technique
    is especially valuable for dermoscopic images where lesions can
    appear at any orientation and scale.
    
    The augmentations used are carefully selected to:
        - Preserve diagnostic features
        - Cover common variations in clinical images
        - Be reversible for spatial consistency
        - Not introduce unrealistic distortions
    
    Standard augmentations include:
        - Horizontal and vertical flips
        - 90-degree rotations
        - Minor scale variations
    
    Attributes:
        n_augmentations (int): Number of augmented versions
        augmentations (List): List of augmentation functions
    
    Note: Computational cost increases linearly with n_augmentations
    """
    
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
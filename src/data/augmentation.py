"""Data augmentation module for skin lesion images.

This module provides comprehensive augmentation pipelines specifically
designed for dermoscopic images. The augmentations help improve model
generalization by simulating various imaging conditions and variations
found in clinical practice.

Key augmentations:
    - Geometric: rotation, flips, affine transformations
    - Color: brightness, contrast, saturation adjustments
    - Noise: Gaussian noise, blur effects
    - Dropout: Coarse dropout for occlusion robustness
    - Test-time augmentation: Multiple predictions for confidence

The augmentations are carefully chosen to:
    1. Preserve diagnostic features
    2. Simulate real-world variations
    3. Handle class imbalance through data diversity
    4. Avoid unrealistic transformations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Optional


class AugmentationFactory:
    """Factory class for creating augmentation pipelines.
    
    Provides static methods to create consistent augmentation
    pipelines for different stages of model development:
    - Training: Heavy augmentation for regularization
    - Validation: Minimal augmentation (only necessary preprocessing)
    - Testing: No augmentation (except normalization)
    - TTA: Multiple augmented versions for ensemble predictions
    
    All pipelines use ImageNet normalization statistics for
    compatibility with pretrained models.
    """

    @staticmethod
    def get_train_transforms(image_size: int = 224) -> A.Compose:
        """Create comprehensive training augmentation pipeline.
        
        Applies various augmentations to increase data diversity and
        improve model generalization. The pipeline includes geometric,
        color, and noise augmentations carefully tuned for dermoscopic images.
        
        Args:
            image_size: Target image size (default: 224)
        
        Returns:
            A.Compose: Albumentations composition pipeline including:
                - Random resized crop (80-100% of original)
                - Rotation (±30°)
                - Horizontal/vertical flips
                - Color jitter variations
                - Gaussian noise/blur
                - Coarse dropout (simulates occlusions)
                - ImageNet normalization
                - Tensor conversion
        
        Note:
            Augmentation probabilities are tuned based on empirical
            results on the HAM10000 dataset.
        """
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(image_size, image_size), scale=(0.8, 1.0)
                ),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MedianBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.3,
                ),
                A.Affine(
                    translate_percent={"x": 0.1, "y": 0.1},
                    scale=(0.9, 1.1),
                    rotate=15,
                    p=0.5
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def get_val_transforms(image_size: int = 224) -> A.Compose:
        """Create validation/test augmentation pipeline.
        
        Minimal augmentation for validation and test sets. Only applies
        necessary preprocessing: resizing and normalization.
        
        Args:
            image_size: Target image size (default: 224)
        
        Returns:
            A.Compose: Minimal pipeline with:
                - Center crop to target size
                - ImageNet normalization
                - Tensor conversion
        """
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
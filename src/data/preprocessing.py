"""Image preprocessing utilities for skin lesion classification.

This module provides various preprocessing functions for dermoscopic images,
including resizing, normalization, hair removal, contrast enhancement, and
other transformations specific to skin lesion analysis.

Key functionalities:
    - Image resizing and normalization
    - Hair artifact removal using morphological operations
    - Contrast enhancement with CLAHE
    - Circular masking for lesion focus
    - Tensor/image conversion utilities

Typical usage example:
    image = cv2.imread("lesion.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(
        image,
        target_size=(224, 224),
        remove_hair_artifacts=True,
        enhance=True
    )
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import torch


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size using bilinear interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C)
        size: Target size as (width, height)
    
    Returns:
        np.ndarray: Resized image with shape (height, width, channels)
    
    Example:
        >>> img = np.random.rand(600, 450, 3)
        >>> resized = resize_image(img, (224, 224))
        >>> resized.shape
        (224, 224, 3)
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range.
    
    Converts uint8 image (0-255) to float32 (0-1) for neural network input.
    
    Args:
        image: Input image with pixel values in [0, 255]
    
    Returns:
        np.ndarray: Normalized image with values in [0, 1]
    
    Example:
        >>> img = np.array([[0, 128, 255]], dtype=np.uint8)
        >>> normalized = normalize_image(img)
        >>> normalized
        array([[0.   , 0.502, 1.   ]], dtype=float32)
    """
    return image.astype(np.float32) / 255.0


def standardize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Standardize image using ImageNet statistics for transfer learning.
    
    Applies z-score normalization with ImageNet's channel-wise mean and std.
    This is required when using pretrained models from torchvision.
    
    Args:
        image: Input image with values in [0, 255]
        mean: Per-channel mean values for normalization (default: ImageNet)
        std: Per-channel standard deviation values (default: ImageNet)
    
    Returns:
        np.ndarray: Standardized image ready for pretrained model input
    
    Note:
        The default mean and std values are from ImageNet dataset and should
        be used when working with models pretrained on ImageNet.
    """
    image = normalize_image(image)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return (image - mean) / std


def remove_hair(image: np.ndarray) -> np.ndarray:
    """Remove hair artifacts from dermoscopy images using morphological operations.
    
    Hair artifacts are common in dermoscopic images and can interfere with
    lesion analysis. This function uses black-hat morphology to detect hair
    structures and inpainting to remove them.
    
    Args:
        image: Input dermoscopic image as numpy array (H, W, C)
    
    Returns:
        np.ndarray: Image with hair artifacts removed
    
    Algorithm:
        1. Convert to grayscale for hair detection
        2. Apply black-hat morphology to isolate hair structures
        3. Create binary mask of hair regions
        4. Use inpainting to fill hair regions with surrounding skin texture
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a kernel for morphological operations
    # Size 17x17 is empirically chosen for typical hair width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Perform blackhat operation to find hair-like structures
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Create binary mask of hair regions
    # Threshold of 10 works well for most images
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the hair regions using Telea's method
    result = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    
    return result


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE improves local contrast and enhances lesion boundaries, making
    features more distinguishable for both human inspection and model training.
    
    Args:
        image: Input image as numpy array (H, W, C)
        clip_limit: Threshold for contrast limiting (default: 2.0)
            Higher values give more contrast but may amplify noise
    
    Returns:
        np.ndarray: Contrast-enhanced image
    
    Note:
        Processing is done in LAB color space to preserve color information
        while enhancing luminance contrast.
    """
    # Convert to LAB color space for luminance-only enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L (luminance) channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result


def crop_center(image: np.ndarray, crop_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Crop the center region of an image.
    
    Center cropping helps focus on the lesion which is typically centered
    in dermoscopic images, removing irrelevant border regions.
    
    Args:
        image: Input image as numpy array (H, W, C)
        crop_size: Size of the crop as (height, width)
            If None, crops to largest centered square
    
    Returns:
        np.ndarray: Center-cropped image
    
    Example:
        >>> img = np.ones((300, 400, 3))
        >>> cropped = crop_center(img, (200, 200))
        >>> cropped.shape
        (200, 200, 3)
    """
    h, w = image.shape[:2]
    
    if crop_size is None:
        # Default to square crop of minimum dimension
        size = min(h, w)
        crop_size = (size, size)
    
    crop_h, crop_w = crop_size
    
    # Calculate center crop coordinates
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[start_h:start_h + crop_h, start_w:start_w + crop_w]


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """Apply a circular mask to focus on the central lesion area.
    
    Many dermoscopic images have a circular field of view. This function
    creates a circular mask to remove corner artifacts and focus the model's
    attention on the relevant circular region.
    
    Args:
        image: Input image as numpy array (H, W, C)
    
    Returns:
        np.ndarray: Image with circular mask applied, corners filled with mean color
    
    Note:
        The background outside the circle is filled with the mean color
        of the image to minimize boundary artifacts.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    
    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask to get circular region
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Calculate mean color of the masked region
    mean_color = cv2.mean(image, mask=mask)[:3]
    
    # Create background with mean color
    background = np.full_like(image, mean_color, dtype=np.uint8)
    
    # Invert mask for background
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=inv_mask)
    
    # Combine foreground and background
    result = cv2.add(result, background)
    
    return result


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    remove_hair_artifacts: bool = False,
    enhance: bool = False,
    apply_mask: bool = False
) -> np.ndarray:
    """Complete preprocessing pipeline for dermoscopic images.
    
    Applies a sequence of preprocessing steps optimized for skin lesion
    classification. The pipeline order is important: hair removal should
    come before enhancement to avoid amplifying artifacts.
    
    Args:
        image: Input dermoscopic image as numpy array (H, W, C)
        target_size: Target size for resizing as (width, height) (default: (224, 224))
        remove_hair_artifacts: Whether to apply hair removal (default: False)
        enhance: Whether to apply contrast enhancement (default: False)
        apply_mask: Whether to apply circular masking (default: False)
    
    Returns:
        np.ndarray: Preprocessed image ready for model input
    
    Example:
        >>> image = cv2.imread("lesion.jpg")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> processed = preprocess_image(
        ...     image,
        ...     target_size=(224, 224),
        ...     remove_hair_artifacts=True,
        ...     enhance=True
        ... )
        >>> processed.shape
        (224, 224, 3)
    """
    # Remove hair artifacts first (if present)
    if remove_hair_artifacts:
        image = remove_hair(image)
    
    # Enhance contrast after hair removal
    if enhance:
        image = enhance_contrast(image)
    
    # Apply circular mask to focus on lesion
    if apply_mask:
        image = apply_circular_mask(image)
    
    # Resize to target size for model input
    image = resize_image(image, target_size)
    
    return image


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """Convert a PyTorch tensor back to image format for visualization.
    
    Reverses the preprocessing steps to convert model input/output tensors
    back to displayable images. Useful for visualization and debugging.
    
    Args:
        tensor: Image tensor of shape (C, H, W) in CHW format
        denormalize: Whether to reverse ImageNet normalization (default: True)
    
    Returns:
        np.ndarray: Image array of shape (H, W, C) with values in [0, 255]
    
    Example:
        >>> tensor = torch.randn(3, 224, 224)
        >>> image = tensor_to_image(tensor)
        >>> image.shape
        (224, 224, 3)
        >>> image.dtype
        dtype('uint8')
    """
    # Move to CPU and convert to numpy
    image = tensor.cpu().numpy()
    
    # Transpose from CHW to HWC format
    image = np.transpose(image, (1, 2, 0))
    
    # Denormalize using ImageNet statistics if requested
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
    
    # Convert to uint8 range [0, 255]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image
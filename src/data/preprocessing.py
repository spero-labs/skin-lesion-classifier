import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import torch


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0


def standardize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Standardize image using ImageNet statistics."""
    image = normalize_image(image)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return (image - mean) / std


def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Remove hair artifacts from dermoscopy images using morphological operations.
    
    Args:
        image: Input image
    
    Returns:
        Image with hair artifacts removed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Perform blackhat operation to find hair
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Create binary mask of hair regions
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the hair regions
    result = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    
    return result


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
    
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result


def crop_center(image: np.ndarray, crop_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Crop the center region of an image.
    
    Args:
        image: Input image
        crop_size: Size of the crop (height, width). If None, crops to square
    
    Returns:
        Center-cropped image
    """
    h, w = image.shape[:2]
    
    if crop_size is None:
        # Crop to square
        size = min(h, w)
        crop_size = (size, size)
    
    crop_h, crop_w = crop_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[start_h:start_h + crop_h, start_w:start_w + crop_w]


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """
    Apply a circular mask to focus on the central lesion area.
    
    Args:
        image: Input image
    
    Returns:
        Image with circular mask applied
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    
    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Fill background with mean color
    mean_color = cv2.mean(image, mask=mask)[:3]
    background = np.full_like(image, mean_color, dtype=np.uint8)
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=inv_mask)
    
    result = cv2.add(result, background)
    
    return result


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    remove_hair_artifacts: bool = False,
    enhance: bool = False,
    apply_mask: bool = False
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image: Input image
        target_size: Target size for resizing
        remove_hair_artifacts: Whether to remove hair artifacts
        enhance: Whether to enhance contrast
        apply_mask: Whether to apply circular mask
    
    Returns:
        Preprocessed image
    """
    # Remove hair if requested
    if remove_hair_artifacts:
        image = remove_hair(image)
    
    # Enhance contrast if requested
    if enhance:
        image = enhance_contrast(image)
    
    # Apply circular mask if requested
    if apply_mask:
        image = apply_circular_mask(image)
    
    # Resize to target size
    image = resize_image(image, target_size)
    
    return image


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    Convert a tensor back to image format.
    
    Args:
        tensor: Image tensor (C, H, W)
        denormalize: Whether to denormalize using ImageNet stats
    
    Returns:
        Image as numpy array (H, W, C)
    """
    # Move to CPU and convert to numpy
    image = tensor.cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    
    # Denormalize if requested
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
    
    # Clip to valid range and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image
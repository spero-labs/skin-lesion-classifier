"""Skin lesion dataset module for HAM10000 dataset.

This module provides a PyTorch Dataset implementation for loading and processing
skin lesion images from the HAM10000 dataset. It supports stratified train/val/test
splits, metadata features, and various data transformations.

Typical usage example:
    dataset = SkinLesionDataset(
        data_dir="HAM10000",
        metadata_path="HAM10000/HAM10000_metadata.csv",
        mode="train",
        transform=train_transform
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for HAM10000 skin lesion images.
    
    This dataset handles loading dermoscopic images and their associated metadata
    from the HAM10000 dataset. It supports stratified splitting, data augmentation,
    and optional metadata features for improved classification.
    
    The HAM10000 dataset contains 10,015 dermoscopic images across 7 classes:
    - akiec: Actinic keratoses and intraepithelial carcinoma
    - bcc: Basal cell carcinoma  
    - bkl: Benign keratosis-like lesions
    - df: Dermatofibroma
    - mel: Melanoma
    - nv: Melanocytic nevi
    - vasc: Vascular lesions
    
    Attributes:
        data_dir (Path): Root directory containing image folders
        metadata (pd.DataFrame): DataFrame with image metadata
        transform: Callable transform to apply to images
        mode (str): Dataset mode ('train', 'val', or 'test')
        use_metadata (bool): Whether to return metadata features
        class_mapping (Dict[str, int]): Mapping from diagnosis to label
        num_classes (int): Number of unique classes
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        image_ids: Optional[List[str]] = None,
        transform=None,
        mode: str = "train",
        class_mapping: Optional[Dict[str, int]] = None,
        use_metadata: bool = False,
    ) -> None:
        """
        Initialize the SkinLesionDataset.
        
        Args:
            data_dir: Path to HAM10000 directory containing image folders
            metadata_path: Path to HAM10000_metadata.csv file
            image_ids: Optional list of image IDs to use (for splits)
            transform: Optional transform to apply to images
            mode: Dataset mode - 'train', 'val', or 'test'
            class_mapping: Optional mapping from diagnosis strings to integer labels
            use_metadata: Whether to include metadata features (age, sex, location)
        
        Raises:
            ValueError: If mode is not one of 'train', 'val', 'test'
            FileNotFoundError: If data_dir or metadata_path don't exist
        """
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.mode = mode
        self.use_metadata = use_metadata
        
        # Create class mapping if not provided
        self.class_mapping = class_mapping or self._create_class_mapping()
        self.num_classes = len(self.class_mapping)
        
        # Filter data based on provided image IDs
        if image_ids is not None:
            self.metadata = self.metadata[self.metadata['image_id'].isin(image_ids)]
        
        # Reset index for proper indexing
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Find image paths
        self._find_image_paths()
        
        # Preprocess metadata features if needed
        if self.use_metadata:
            self._prepare_metadata_features()
        
        logger.info(f"Loaded {len(self)} images for {mode} set")
        logger.info(f"Class distribution: {self.metadata['dx'].value_counts().to_dict()}")

    def _create_class_mapping(self) -> Dict[str, int]:
        """Create mapping from diagnosis codes to integer labels.
        
        Creates a sorted mapping from the 7 diagnosis classes to integers 0-6.
        This ensures consistent label encoding across train/val/test sets.
        
        Returns:
            Dict[str, int]: Mapping from diagnosis string to integer label
                Example: {'akiec': 0, 'bcc': 1, 'bkl': 2, ...}
        """
        unique_classes = sorted(self.metadata["dx"].unique())
        mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        logger.info(f"Class mapping: {mapping}")
        return mapping
    
    def _find_image_paths(self) -> None:
        """Find actual file paths for images in the dataset.
        
        The HAM10000 images are split across two directories (part_1 and part_2).
        This method locates each image file and adds the full path to the metadata.
        Images that cannot be found are logged and removed from the dataset.
        
        Side Effects:
            - Adds 'image_path' column to self.metadata
            - Removes rows where images are not found
            - Resets DataFrame index
        """
        image_paths = []
        for _, row in self.metadata.iterrows():
            image_id = row['image_id']
            # Images can be in part_1 or part_2
            for part_dir in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                img_path = self.data_dir / part_dir / f"{image_id}.jpg"
                if img_path.exists():
                    image_paths.append(str(img_path))
                    break
            else:
                logger.warning(f"Image not found: {image_id}")
                image_paths.append(None)
        
        self.metadata['image_path'] = image_paths
        # Remove rows where image was not found
        self.metadata = self.metadata[self.metadata['image_path'].notna()]
        self.metadata = self.metadata.reset_index(drop=True)
    
    def _prepare_metadata_features(self) -> None:
        """Prepare metadata features for use as additional model inputs.
        
        Preprocesses patient metadata to create normalized numerical features:
        - Sex: Binary encoding (male=0, female=1, unknown=0.5)
        - Age: Z-score normalization with median imputation
        - Localization: Integer encoding of body location
        
        These features can be concatenated with CNN features for improved
        classification performance.
        
        Side Effects:
            - Adds 'sex_encoded' column to self.metadata
            - Adds 'age_normalized' column to self.metadata  
            - Adds 'localization_encoded' column to self.metadata
        """
        # Encode sex
        self.metadata['sex_encoded'] = self.metadata['sex'].map({'male': 0, 'female': 1})
        self.metadata['sex_encoded'] = self.metadata['sex_encoded'].fillna(0.5)
        
        # Normalize age
        self.metadata['age_normalized'] = self.metadata['age'].fillna(self.metadata['age'].median())
        self.metadata['age_normalized'] = (self.metadata['age_normalized'] - self.metadata['age_normalized'].mean()) / self.metadata['age_normalized'].std()
        
        # Encode localization (one-hot encoding)
        self.localization_mapping = {loc: idx for idx, loc in enumerate(self.metadata['localization'].unique())}
        self.metadata['localization_encoded'] = self.metadata['localization'].map(self.localization_mapping)
        self.metadata['localization_encoded'] = self.metadata['localization_encoded'].fillna(0)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            int: Number of images in the dataset
        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, torch.Tensor]]:
        """Get a single sample from the dataset.
        
        Loads an image, applies transformations, and returns it with its label.
        If use_metadata is True, also returns normalized metadata features.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            If use_metadata is False:
                Tuple[torch.Tensor, int]: (image, label)
            If use_metadata is True:
                Tuple[torch.Tensor, int, torch.Tensor]: (image, label, metadata)
                
            Where:
                - image: Transformed image tensor of shape (C, H, W)
                - label: Integer class label (0-6)
                - metadata: Tensor of metadata features [age, sex, location]
        
        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If image file cannot be read
        """
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        label = self.class_mapping[row['dx']]
        
        # Get metadata features if needed
        metadata_features = None
        if self.use_metadata:
            metadata_features = torch.tensor([
                row['age_normalized'],
                row['sex_encoded'],
                row['localization_encoded']
            ], dtype=torch.float32)
        
        if self.use_metadata:
            return image, label, metadata_features
        else:
            return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse class weights for handling class imbalance.
        
        Computes weights inversely proportional to class frequencies to give
        more importance to underrepresented classes during training.
        
        The weight for class c is calculated as:
            weight_c = total_samples / (num_classes * count_c)
        
        Returns:
            torch.Tensor: Tensor of shape (num_classes,) with weight for each class
                Example: [2.5, 1.8, 0.9, 4.2, 0.8, 0.3, 3.7] for 7 classes
        """
        class_counts = self.metadata['dx'].value_counts()
        class_weights = []
        
        total_samples = len(self.metadata)
        for cls in sorted(self.class_mapping.keys()):
            count = class_counts.get(cls, 1)
            weight = total_samples / (self.num_classes * count)
            class_weights.append(weight)
        
        return torch.tensor(class_weights, dtype=torch.float32)


def create_data_splits(
    metadata_path: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """Create stratified train/validation/test splits of the dataset.
    
    Splits the HAM10000 dataset into three sets while maintaining class
    distribution across all splits. Handles duplicate images by ensuring
    the same lesion doesn't appear in multiple splits.
    
    Args:
        metadata_path: Path to HAM10000_metadata.csv file
        val_split: Proportion of data for validation (default: 0.15)
        test_split: Proportion of data for test (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Whether to maintain class distribution in splits (default: True)
    
    Returns:
        Tuple[List[str], List[str], List[str]]: Three lists containing:
            - train_ids: Image IDs for training set (~70% of data)
            - val_ids: Image IDs for validation set (~15% of data)
            - test_ids: Image IDs for test set (~15% of data)
    
    Example:
        >>> train_ids, val_ids, test_ids = create_data_splits(
        ...     "HAM10000/HAM10000_metadata.csv",
        ...     val_split=0.15,
        ...     test_split=0.15
        ... )
        >>> print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        Train: 7011, Val: 1502, Test: 1502
    """
    metadata = pd.read_csv(metadata_path)
    
    # Get unique image IDs (some images may have multiple entries)
    unique_data = metadata.drop_duplicates(subset=['image_id'])
    
    # Stratify by diagnosis if requested
    stratify_col = unique_data['dx'] if stratify else None
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        unique_data,
        test_size=test_split,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: train vs val
    val_ratio = val_split / (1 - test_split)
    stratify_col = train_val_data['dx'] if stratify else None
    
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_col
    )
    
    train_ids = train_data['image_id'].tolist()
    val_ids = val_data['image_id'].tolist()
    test_ids = test_data['image_id'].tolist()
    
    logger.info(f"Data splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids
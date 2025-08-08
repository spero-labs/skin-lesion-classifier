import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class SkinLesionDataset(Dataset):
    """Custom dataset for skin lesion images."""

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        image_ids: Optional[List[str]] = None,
        transform=None,
        mode: str = "train",
        class_mapping: Optional[Dict] = None,
        use_metadata: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to HAM10000 directory
            metadata_path: Path to metadata CSV file
            image_ids: List of image IDs to use (for train/val/test splits)
            transform: Image transformations
            mode: One of 'train', 'val', 'test'
            class_mapping: Mapping from diagnosis to integer labels
            use_metadata: Whether to include metadata features
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

    def _create_class_mapping(self) -> Dict:
        """Create mapping from diagnosis to integer labels."""
        unique_classes = sorted(self.metadata["dx"].unique())
        mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        logger.info(f"Class mapping: {mapping}")
        return mapping
    
    def _find_image_paths(self):
        """Find actual paths for images."""
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
    
    def _prepare_metadata_features(self):
        """Prepare metadata features for model input."""
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
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (image, label, metadata_features)
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
        """Calculate class weights for handling imbalance."""
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
    """
    Create train/val/test splits of image IDs.
    
    Args:
        metadata_path: Path to metadata CSV
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random seed
        stratify: Whether to stratify by diagnosis
    
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
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
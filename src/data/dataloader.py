import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional, Dict
from .dataset import SkinLesionDataset, create_data_splits
from .augmentation import AugmentationFactory
import logging

logger = logging.getLogger(__name__)


class DataModule:
    """Data module for managing train/val/test dataloaders."""
    
    def __init__(self, config: Dict):
        """
        Initialize data module.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'HAM10000')
        self.metadata_path = config.get('metadata_path', 'HAM10000/HAM10000_metadata.csv')
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.image_size = config.get('image_size', 224)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        self.use_metadata = config.get('use_metadata', False)
        self.use_weighted_sampling = config.get('use_weighted_sampling', True)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_mapping = None
        self.num_classes = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for train/val/test."""
        # Create data splits
        train_ids, val_ids, test_ids = create_data_splits(
            self.metadata_path,
            val_split=self.val_split,
            test_split=self.test_split,
            random_state=self.config.get('seed', 42),
            stratify=True
        )
        
        # Get augmentations
        train_transform = AugmentationFactory.get_train_transforms(self.image_size)
        val_transform = AugmentationFactory.get_val_transforms(self.image_size)
        
        # Create datasets
        self.train_dataset = SkinLesionDataset(
            data_dir=self.data_dir,
            metadata_path=self.metadata_path,
            image_ids=train_ids,
            transform=train_transform,
            mode='train',
            use_metadata=self.use_metadata
        )
        
        self.val_dataset = SkinLesionDataset(
            data_dir=self.data_dir,
            metadata_path=self.metadata_path,
            image_ids=val_ids,
            transform=val_transform,
            mode='val',
            class_mapping=self.train_dataset.class_mapping,
            use_metadata=self.use_metadata
        )
        
        self.test_dataset = SkinLesionDataset(
            data_dir=self.data_dir,
            metadata_path=self.metadata_path,
            image_ids=test_ids,
            transform=val_transform,
            mode='test',
            class_mapping=self.train_dataset.class_mapping,
            use_metadata=self.use_metadata
        )
        
        self.class_mapping = self.train_dataset.class_mapping
        self.num_classes = self.train_dataset.num_classes
        
        logger.info(f"Dataset setup complete. Classes: {self.num_classes}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader with optional weighted sampling."""
        sampler = None
        shuffle = True
        
        if self.use_weighted_sampling:
            # Create weighted sampler for handling class imbalance
            class_weights = self.train_dataset.get_class_weights()
            sample_weights = []
            
            for idx in range(len(self.train_dataset)):
                label = self.train_dataset.metadata.iloc[idx]['dx']
                class_idx = self.train_dataset.class_mapping[label]
                sample_weights.append(class_weights[class_idx])
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # Sampler handles shuffling
        
        # Check if MPS is being used
        import torch
        pin_memory = torch.cuda.is_available()  # Only pin memory for CUDA
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        import torch
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        import torch
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function."""
        if self.train_dataset is None:
            self.setup()
        return self.train_dataset.get_class_weights()
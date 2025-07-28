from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


class DataModule:
    """Data module handling all data operations"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Setup datasets with proper splits"""
        # Create datasets
        # Calculate class weights for imbalanced data
        pass

    def get_weighted_sampler(self, dataset):
        """Create weighted sampler for balanced training"""
        class_counts = np.bincount(dataset.labels)
        class_weights = 1.0 / class_counts
        weights = class_weights[dataset.labels]
        return WeightedRandomSampler(weights, len(weights))

    def train_dataloader(self) -> DataLoader:
        sampler = self.get_weighted_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

"""Configuration management module for training and inference.

This module provides dataclass-based configuration management for the
skin lesion classification system. It defines structured configuration
classes for different components and supports loading from YAML files.

Key components:
    - DataConfig: Dataset and data loading settings
    - ModelConfig: Model architecture and hyperparameters
    - TrainingConfig: Training process configuration
    - Config: Main configuration aggregating all components

The configuration system ensures:
    - Type safety with dataclasses
    - Default values for all parameters
    - Easy serialization/deserialization
    - Centralized parameter management

Typical usage:
    # Load from YAML
    config = Config.from_yaml('configs/config.yaml')
    
    # Access nested configurations
    batch_size = config.data.batch_size
    learning_rate = config.training.learning_rate
    
    # Create programmatically
    config = Config(
        data=DataConfig(dataset_path='HAM10000'),
        model=ModelConfig(architecture='efficientnet_b1'),
        training=TrainingConfig(epochs=100)
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.
    
    Defines all parameters related to dataset handling, including
    paths, preprocessing settings, augmentation parameters, and
    data loader configuration.
    
    Attributes:
        dataset_path: Path to dataset directory
        image_size: Target image dimensions (square)
        batch_size: Training batch size
        num_workers: Parallel data loading processes
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        augmentation: Augmentation parameters dictionary
    """
    dataset_path: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    validation_split: float = 0.2
    test_split: float = 0.1
    augmentation: Dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model architecture and hyperparameters.
    
    Defines model selection, architecture-specific parameters,
    and regularization settings.
    
    Attributes:
        architecture: Model name (efficientnet_b0, resnet50, vit_small, etc.)
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes (7 for HAM10000)
        dropout: Dropout rate for regularization
        hidden_size: Hidden layer dimensions in classification head
    """
    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 7
    dropout: float = 0.3
    hidden_size: int = 512


@dataclass
class TrainingConfig:
    """Configuration for training process and optimization.
    
    Defines training hyperparameters, optimization settings,
    and training strategy parameters.
    
    Attributes:
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        scheduler: LR scheduler type (cosine, plateau, step)
        early_stopping_patience: Epochs to wait before stopping
        gradient_clip_val: Maximum gradient norm for clipping
    """
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0


@dataclass
class Config:
    """Main configuration aggregating all component configs.
    
    Central configuration object that combines data, model, and
    training configurations with experiment-level settings.
    
    Attributes:
        data: Data loading and preprocessing configuration
        model: Model architecture configuration
        training: Training process configuration
        experiment_name: Name for experiment tracking
        seed: Random seed for reproducibility
        device: Computing device (cuda, mps, cpu)
    """
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str = "skin_lesion_baseline"
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            Config object with loaded parameters
        
        Example:
            config = Config.from_yaml('configs/experiment.yaml')
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

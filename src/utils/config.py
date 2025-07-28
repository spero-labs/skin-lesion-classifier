from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    dataset_path: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    validation_split: float = 0.2
    test_split: float = 0.1
    augmentation: Dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 7
    dropout: float = 0.3
    hidden_size: int = 512


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str = "skin_lesion_baseline"
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

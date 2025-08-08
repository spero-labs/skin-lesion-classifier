from .logger import setup_logger, get_logger
from .config import Config, DataConfig, ModelConfig, TrainingConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "setup_logger",
    "get_logger",
    "set_seed",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
]
"""Model factory module for centralized model creation and management.

This module provides a factory pattern implementation for creating different
model architectures with consistent configuration. It maintains a registry of
available models and handles the complexity of model instantiation, including
architecture-specific parameters and naming conventions.

The factory pattern provides:
    - Centralized model creation logic
    - Consistent configuration interface
    - Easy addition of new architectures
    - Model name validation and mapping

Supported architectures:
    - EfficientNet family (B0-B3)
    - ResNet with attention (50, 101, 152)
    - Vision Transformers (Small, Base)
    - Swin Transformers (Tiny, Small, Base)
    - Ensemble models

Typical usage:
    model = ModelFactory.create_model(
        architecture='efficientnet_b1',
        num_classes=7,
        pretrained=True,
        dropout=0.3
    )
"""

from typing import Dict, Any
from .architectures import (
    EfficientNetModel,
    ResNetWithAttention,
    VisionTransformerModel,
    SwinTransformerModel,
    EnsembleModel,
)
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and configuring models.
    
    Centralizes model creation logic and provides a consistent interface
    for instantiating different architectures. Handles the mapping between
    user-friendly model names and the actual model implementations.
    
    The factory maintains:
        - Registry of available model classes
        - Mapping of model names to timm model identifiers
        - Validation of model configurations
    
    This design pattern simplifies:
        - Adding new architectures
        - Switching between models
        - Maintaining consistent configuration
        - Model versioning and updates
    """
    
    # Model registry
    _models = {
        "efficientnet_b0": EfficientNetModel,
        "efficientnet_b1": EfficientNetModel,
        "efficientnet_b2": EfficientNetModel,
        "efficientnet_b3": EfficientNetModel,
        "resnet50": ResNetWithAttention,
        "resnet101": ResNetWithAttention,
        "resnet152": ResNetWithAttention,
        "vit_small": VisionTransformerModel,
        "vit_base": VisionTransformerModel,
        "swin_tiny": SwinTransformerModel,
        "swin_small": SwinTransformerModel,
        "swin_base": SwinTransformerModel,
    }
    
    # Model name mappings
    _model_names = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "efficientnet_b2": "efficientnet_b2",
        "efficientnet_b3": "efficientnet_b3",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "resnet152": "resnet152",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
    }
    
    @classmethod
    def create_model(
        cls,
        architecture: str,
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_metadata: bool = False,
        **kwargs
    ) -> BaseModel:
        """
        Create a model based on architecture name.
        
        Args:
            architecture: Model architecture name
            num_classes: Number of output classes
            dropout: Dropout rate
            pretrained: Whether to use pretrained weights
            use_metadata: Whether to use metadata features
            **kwargs: Additional model-specific arguments
        
        Returns:
            Model instance
        """
        if architecture not in cls._models:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Available: {list(cls._models.keys())}")
        
        model_class = cls._models[architecture]
        model_name = cls._model_names[architecture]
        
        # Prepare model arguments
        model_args = {
            "num_classes": num_classes,
            "dropout": dropout,
            "pretrained": pretrained,
            "use_metadata": use_metadata,
        }
        
        # Add model-specific name for architectures that need it
        if architecture.startswith("efficientnet"):
            model_args["model_name"] = model_name
        elif architecture.startswith("resnet"):
            model_args["model_name"] = model_name
        elif architecture.startswith("vit"):
            model_args["model_name"] = model_name
        elif architecture.startswith("swin"):
            model_args["model_name"] = model_name
        
        # Merge with additional kwargs
        model_args.update(kwargs)
        
        # Create model
        model = model_class(**model_args)
        
        logger.info(f"Created {architecture} model with {num_classes} classes")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    @classmethod
    def create_ensemble(
        cls,
        architectures: list,
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_metadata: bool = False,
        weights: list = None,
        **kwargs
    ) -> EnsembleModel:
        """
        Create an ensemble of models.
        
        Args:
            architectures: List of model architecture names
            num_classes: Number of output classes
            dropout: Dropout rate
            pretrained: Whether to use pretrained weights
            use_metadata: Whether to use metadata features
            weights: Optional weights for ensemble members
            **kwargs: Additional model-specific arguments
        
        Returns:
            Ensemble model instance
        """
        if len(architectures) == 0:
            raise ValueError("At least one architecture is required for ensemble")
        
        models = []
        for arch in architectures:
            model = cls.create_model(
                arch,
                num_classes=num_classes,
                dropout=dropout,
                pretrained=pretrained,
                use_metadata=use_metadata,
                **kwargs
            )
            models.append(model)
        
        ensemble = EnsembleModel(models, weights)
        
        logger.info(f"Created ensemble with {len(models)} models: {architectures}")
        logger.info(f"Ensemble parameters: {sum(p.numel() for p in ensemble.parameters()):,}")
        
        return ensemble
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model architectures."""
        return list(cls._models.keys())
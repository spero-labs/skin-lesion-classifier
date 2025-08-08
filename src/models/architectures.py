"""Model architectures module for skin lesion classification.

This module provides state-of-the-art deep learning architectures optimized
for medical image classification. Each architecture is designed to handle the
specific challenges of dermoscopic image analysis including fine-grained
features, class imbalance, and limited training data.

Key architectures:
    - EfficientNet: Efficient scaling with compound coefficient
    - ResNet with Attention: Deep residual learning with attention mechanisms
    - Vision Transformer (ViT): Pure transformer architecture for images
    - Swin Transformer: Hierarchical vision transformer with shifted windows
    - Ensemble Model: Combines multiple architectures for robustness

All models support:
    - Transfer learning from ImageNet pretrained weights
    - Custom classification heads with dropout and batch normalization
    - Optional metadata fusion (age, sex, location)
    - Attention mechanisms for feature enhancement
    - Flexible backbone freezing for fine-tuning

Typical usage:
    model = EfficientNetModel(
        model_name='efficientnet_b1',
        num_classes=7,
        pretrained=True,
        use_metadata=True
    )
    
    # For inference
    logits = model(images, metadata)
    
    # For feature extraction
    features = model.get_features(images)
"""

import timm
import torch
import torch.nn as nn
from typing import List, Optional
from .base_model import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet with custom head and attention mechanism.
    
    EfficientNet uses a compound scaling method that uniformly scales
    network width, depth, and resolution with a set of fixed scaling
    coefficients. This architecture is particularly effective for
    medical images due to its excellent accuracy-efficiency trade-off.
    
    Key features:
        - Compound scaling for optimal resource utilization
        - Attention mechanism for focusing on lesion regions
        - Lightweight architecture suitable for deployment
        - Strong performance on small datasets with transfer learning
    
    Architecture details:
        - Backbone: EfficientNet-B0 to B7 (configurable)
        - Attention: Channel-wise attention with sigmoid gating
        - Head: 3-layer MLP with batch normalization and dropout
        - Optional: Metadata fusion for demographic features
    
    Performance characteristics:
        - Model size: 20-60MB depending on variant
        - Inference time: <50ms on GPU
        - Memory usage: 2-4GB during training
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_metadata: bool = False,
        metadata_dim: int = 3,
    ):
        super().__init__(num_classes, dropout)
        
        self.use_metadata = use_metadata
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Metadata fusion if needed
        if use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
            )
            classifier_input_dim = feature_dim + 128
        else:
            classifier_input_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Fuse metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            features = torch.cat([features, metadata_features], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        features = self.backbone(x)
        attention_weights = self.attention(features)
        return features * attention_weights


class ResNetWithAttention(BaseModel):
    """ResNet with dual attention mechanism and custom head.
    
    Enhances the standard ResNet architecture with both channel and
    spatial attention mechanisms. This combination helps the model
    focus on relevant features and spatial regions, crucial for
    identifying subtle dermoscopic patterns.
    
    Key features:
        - Dual attention: Channel attention (SE-like) + Spatial attention
        - Deep residual connections prevent gradient vanishing
        - Robust to various image qualities and conditions
        - Excellent for capturing multi-scale features
    
    Attention mechanisms:
        1. Channel Attention: Recalibrates channel-wise feature responses
           - Squeeze: Global average pooling
           - Excitation: Two FC layers with ReLU
           - Scale: Element-wise multiplication
        
        2. Spatial Attention: Focuses on informative spatial regions
           - Aggregate: Channel-wise max and average pooling
           - Transform: 7x7 convolution
           - Scale: Element-wise multiplication
    
    Architecture variants:
        - ResNet18: Lightweight, good for quick experiments
        - ResNet34: Balanced performance
        - ResNet50: Standard choice, good accuracy
        - ResNet101: High capacity for complex patterns
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_metadata: bool = False,
        metadata_dim: int = 3,
    ):
        super().__init__(num_classes, dropout)
        
        self.use_metadata = use_metadata
        
        # Load pretrained ResNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        feature_dim = self.backbone.num_features
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Metadata fusion
        if use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
            )
            classifier_input_dim = feature_dim + 128
        else:
            classifier_input_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract backbone features (before GAP)
        # We need to modify this to get intermediate features
        # For ResNet, we'll extract features manually
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Apply spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fuse metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            x = torch.cat([x, metadata_features], dim=1)
        
        # Classification
        output = self.classifier(x)
        return output


class VisionTransformerModel(BaseModel):
    """Vision Transformer (ViT) for skin lesion classification.
    
    Applies the transformer architecture directly to sequences of image
    patches, treating an image as a sequence of visual tokens. This
    approach captures long-range dependencies and global context better
    than convolutional networks.
    
    Key features:
        - Pure transformer architecture without convolutions
        - Global receptive field from the first layer
        - Position embeddings for spatial information
        - Strong performance with sufficient data and augmentation
    
    How it works:
        1. Divide image into fixed-size patches (16x16 or 32x32)
        2. Linearly embed each patch
        3. Add position embeddings
        4. Process with transformer encoder blocks
        5. Use [CLS] token or average pooling for classification
    
    Advantages for medical imaging:
        - Captures global context important for diagnosis
        - Less inductive bias, learns from data
        - Excellent transfer learning from large datasets
        - Interpretable attention maps
    
    Model variants:
        - ViT-Tiny: 5.7M parameters, fast inference
        - ViT-Small: 22M parameters, good balance
        - ViT-Base: 86M parameters, high accuracy
        - ViT-Large: 307M parameters, state-of-the-art
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_metadata: bool = False,
        metadata_dim: int = 3,
    ):
        super().__init__(num_classes, dropout)
        
        self.use_metadata = use_metadata
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        feature_dim = self.backbone.num_features
        
        # Metadata fusion
        if use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
            )
            classifier_input_dim = feature_dim + 128
        else:
            classifier_input_dim = feature_dim
        
        # Custom classification head with more capacity
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Fuse metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            features = torch.cat([features, metadata_features], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output


class SwinTransformerModel(BaseModel):
    """Swin Transformer for hierarchical skin lesion classification.
    
    Swin (Shifted Window) Transformer builds hierarchical feature maps
    and has linear computational complexity with respect to image size.
    It combines the strengths of CNNs (hierarchical features, translation
    invariance) with transformers (long-range dependencies, flexibility).
    
    Key innovations:
        - Hierarchical architecture with merging layers
        - Shifted window approach for cross-window connections
        - Linear complexity O(n) instead of quadratic O(n²)
        - Multi-scale feature representations
    
    Architecture details:
        - Patch partition: 4x4 patches as input tokens
        - 4 stages with progressively merged patches
        - Window-based self-attention (7x7 windows)
        - Shifted windows for cross-window connections
    
    Advantages for dermoscopy:
        - Captures both local textures and global structure
        - Efficient for high-resolution medical images
        - Strong performance on various scales
        - Better than ViT on smaller datasets
    
    Model variants:
        - Swin-Tiny: 28M params, efficient
        - Swin-Small: 50M params, balanced
        - Swin-Base: 88M params, high performance
        - Swin-Large: 197M params, maximum accuracy
    """
    
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_metadata: bool = False,
        metadata_dim: int = 3,
    ):
        super().__init__(num_classes, dropout)
        
        self.use_metadata = use_metadata
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        feature_dim = self.backbone.num_features
        
        # Metadata fusion
        if use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
            )
            classifier_input_dim = feature_dim + 128
        else:
            classifier_input_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Fuse metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            features = torch.cat([features, metadata_features], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output


class EnsembleModel(BaseModel):
    """Ensemble of multiple models for improved robustness and accuracy.
    
    Combines predictions from multiple diverse models to achieve better
    performance than any individual model. Particularly effective for
    medical imaging where different architectures capture complementary
    features and reduce prediction variance.
    
    Ensemble strategies:
        - Weighted averaging: Combine logits with learned weights
        - Voting: Hard or soft voting across models
        - Stacking: Meta-learner on top of base models
    
    Benefits:
        - Reduced overfitting through model diversity
        - Improved generalization to unseen data
        - Higher confidence in predictions
        - Robustness to model-specific failures
    
    Typical ensemble composition:
        1. EfficientNet: Efficient feature extraction
        2. ResNet+Attention: Spatial pattern recognition
        3. Transformer: Global context understanding
    
    Performance gains:
        - Typically 2-5% improvement in AUC-ROC
        - More stable predictions across datasets
        - Better calibrated probability estimates
    
    Note: Increases inference time proportionally to number of models
    """

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        if len(models) == 0:
            raise ValueError("At least one model is required for ensemble")
        
        super().__init__(models[0].num_classes, models[0].dropout)
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = weights

    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = []
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'use_metadata') and model.use_metadata:
                output = model(x, metadata)
            else:
                output = model(x)
            outputs.append(weight * output)
        
        return torch.stack(outputs).sum(dim=0)
    
    def freeze_backbone(self):
        """Freeze all model backbones."""
        for model in self.models:
            model.freeze_backbone()
    
    def unfreeze_backbone(self):
        """Unfreeze all model backbones."""
        for model in self.models:
            model.unfreeze_backbone()
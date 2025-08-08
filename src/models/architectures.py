import timm
import torch
import torch.nn as nn
from typing import List, Optional
from .base_model import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet with custom head and attention mechanism."""

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
    """ResNet with attention mechanism and custom head."""
    
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
    """Vision Transformer for skin lesion classification."""

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
    """Swin Transformer for skin lesion classification."""
    
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
    """Ensemble of multiple models for improved performance."""

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
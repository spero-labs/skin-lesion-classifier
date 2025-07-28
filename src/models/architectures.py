import timm
import torch
import torch.nn as nn
from .base_model import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet with custom head"""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 7,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__(num_classes, dropout)

        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        feature_dim = self.backbone.num_features

        # Custom head with attention
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)

        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights

        # Classification
        output = self.classifier(features)
        return output


class VisionTransformerModel(BaseModel):
    """Vision Transformer for skin lesion classification"""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3):
        super().__init__(num_classes, dropout)
        # ViT implementation
        pass


class EnsembleModel(BaseModel):
    """Ensemble of multiple models"""

    def __init__(self, models: List[BaseModel]):
        super().__init__(models[0].num_classes)
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

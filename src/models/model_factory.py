from typing import Dict, Any
from .architectures import EfficientNetModel, VisionTransformerModel


class ModelFactory:
    """Factory for creating models"""

    _models = {
        "efficientnet_b0": EfficientNetModel,
        "efficientnet_b1": EfficientNetModel,
        "vit_base": VisionTransformerModel,
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> BaseModel:
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = cls._models[model_name]
        return model_class(**kwargs)

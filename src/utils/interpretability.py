from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np


class ModelInterpreter:
    """Generate interpretability visualizations"""

    def __init__(self, model: torch.nn.Module, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._get_target_layer()
        self.cam = GradCAM(model=model, target_layers=[self.target_layer])

    def generate_gradcam(
        self, image: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate GradCAM heatmap"""
        grayscale_cam = self.cam(input_tensor=image.unsqueeze(0), targets=class_idx)
        return grayscale_cam[0]

    def visualize_attention(self, model_output):
        """Visualize attention weights if available"""
        pass

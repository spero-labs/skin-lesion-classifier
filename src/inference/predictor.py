import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import cv2
import logging

from ..models import ModelFactory
from ..data.preprocessing import preprocess_image, tensor_to_image
from ..data.augmentation import AugmentationFactory
from .tta import TestTimeAugmentation

logger = logging.getLogger(__name__)


class SkinLesionPredictor:
    """Inference class for skin lesion prediction."""
    
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    CLASS_DESCRIPTIONS = {
        "akiec": "Actinic keratoses and intraepithelial carcinoma",
        "bcc": "Basal cell carcinoma",
        "bkl": "Benign keratosis-like lesions",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic nevi",
        "vasc": "Vascular lesions"
    }
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = None,
        use_tta: bool = False,
        tta_transforms: int = 5
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Optional path to configuration
            device: Device to use
            use_tta: Whether to use test-time augmentation
            tta_transforms: Number of TTA transforms
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tta = use_tta
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = AugmentationFactory.get_val_transforms(224)
        
        # Setup TTA if requested
        if use_tta:
            self.tta = TestTimeAugmentation(n_augmentations=tta_transforms)
        
        logger.info(f"Predictor initialized. Device: {self.device}")
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model architecture from checkpoint
        # This is a simplified version - in production, save config with checkpoint
        model_state = checkpoint['model_state_dict']
        
        # Create model (assuming EfficientNet-B0 for now)
        model = ModelFactory.create_model(
            architecture="efficientnet_b0",
            num_classes=7,
            pretrained=False
        )
        
        model.load_state_dict(model_state)
        model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, Path],
        return_probabilities: bool = True,
        return_top_k: int = 3
    ) -> Dict:
        """
        Predict single image.
        
        Args:
            image_path: Path to image
            return_probabilities: Whether to return class probabilities
            return_top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Get predictions
        if self.use_tta:
            outputs = self.tta(self.model, image_tensor)
        else:
            outputs = self.model(image_tensor)
        
        # Process outputs
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:return_top_k]
        top_k_predictions = []
        
        for idx in top_k_indices:
            top_k_predictions.append({
                "class": self.CLASS_NAMES[idx],
                "description": self.CLASS_DESCRIPTIONS[self.CLASS_NAMES[idx]],
                "probability": float(probs[idx])
            })
        
        result = {
            "predicted_class": self.CLASS_NAMES[pred_class],
            "confidence": float(confidence),
            "top_predictions": top_k_predictions
        }
        
        if return_probabilities:
            result["all_probabilities"] = {
                self.CLASS_NAMES[i]: float(probs[i])
                for i in range(len(self.CLASS_NAMES))
            }
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load and preprocess batch
            for path in batch_paths:
                image = cv2.imread(str(path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = self.transform(image=image)
                batch_images.append(transformed['image'])
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Get predictions
            outputs = self.model(batch_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            # Process each prediction
            for j, prob in enumerate(probs):
                pred_class = np.argmax(prob)
                confidence = prob[pred_class]
                
                results.append({
                    "image_path": str(batch_paths[j]),
                    "predicted_class": self.CLASS_NAMES[pred_class],
                    "confidence": float(confidence),
                    "probabilities": {
                        self.CLASS_NAMES[k]: float(prob[k])
                        for k in range(len(self.CLASS_NAMES))
                    }
                })
        
        return results
    
    def explain_prediction(
        self,
        image_path: Union[str, Path],
        method: str = "gradcam"
    ) -> Tuple[Dict, np.ndarray]:
        """
        Explain prediction with visualization.
        
        Args:
            image_path: Path to image
            method: Explanation method (gradcam, etc.)
        
        Returns:
            Tuple of (prediction dict, visualization array)
        """
        # Get prediction
        prediction = self.predict(image_path)
        
        # Generate explanation visualization
        # This would use GradCAM or other interpretability methods
        # For now, return placeholder
        image = cv2.imread(str(image_path))
        visualization = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return prediction, visualization
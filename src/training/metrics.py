import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
)
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate and track training metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch.
        
        Args:
            preds: Predicted class indices
            targets: True class indices
            probs: Optional prediction probabilities
        """
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to compute metrics")
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "balanced_accuracy": balanced_accuracy_score(targets, preds),
            "cohen_kappa": cohen_kappa_score(targets, preds),
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, labels=list(range(self.num_classes))
        )
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f"{class_name}_precision"] = precision[i]
            metrics[f"{class_name}_recall"] = recall[i]
            metrics[f"{class_name}_f1"] = f1[i]
            metrics[f"{class_name}_support"] = int(support[i])
        
        # Macro and weighted averages
        metrics["macro_precision"] = np.mean(precision)
        metrics["macro_recall"] = np.mean(recall)
        metrics["macro_f1"] = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        metrics["weighted_precision"] = weighted_precision
        metrics["weighted_recall"] = weighted_recall
        metrics["weighted_f1"] = weighted_f1
        
        # AUC-ROC if probabilities are available
        if len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            
            # One-hot encode targets for multi-class AUC
            targets_one_hot = np.zeros((len(targets), self.num_classes))
            targets_one_hot[np.arange(len(targets)), targets] = 1
            
            try:
                # Overall AUC (macro average)
                auc_macro = roc_auc_score(
                    targets_one_hot, probs, average="macro", multi_class="ovr"
                )
                metrics["auc_macro"] = auc_macro
                
                # Per-class AUC
                for i, class_name in enumerate(self.class_names):
                    if len(np.unique(targets_one_hot[:, i])) > 1:  # Need both classes
                        auc = roc_auc_score(targets_one_hot[:, i], probs[:, i])
                        metrics[f"{class_name}_auc"] = auc
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
        
        # Sensitivity (Recall) and Specificity for binary or main classes
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f"{class_name}_sensitivity"] = sensitivity
            metrics[f"{class_name}_specificity"] = specificity
        
        # Average sensitivity and specificity
        avg_sensitivity = np.mean([metrics[f"{cn}_sensitivity"] for cn in self.class_names])
        avg_specificity = np.mean([metrics[f"{cn}_specificity"] for cn in self.class_names])
        
        metrics["avg_sensitivity"] = avg_sensitivity
        metrics["avg_specificity"] = avg_specificity
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if len(self.predictions) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(
            self.targets,
            self.predictions,
            labels=list(range(self.num_classes))
        )
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if len(self.predictions) == 0:
            return "No predictions available"
        
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            digits=4
        )


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    probabilities: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate metrics for a single batch or epoch.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        num_classes: Number of classes
        probabilities: Optional prediction probabilities
    
    Returns:
        Dictionary of metrics
    """
    calculator = MetricsCalculator(num_classes)
    calculator.update(predictions, targets, probabilities)
    return calculator.compute()
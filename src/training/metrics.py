from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import numpy as np
from typing import Dict, List


class MetricsCalculator:
    """Calculate and track metrics"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []
        self.probabilities = []

    def update(self, preds: np.ndarray, labels: np.ndarray, probs: np.ndarray):
        self.predictions.extend(preds)
        self.labels.extend(labels)
        self.probabilities.extend(probs)

    def compute_metrics(self) -> Dict:
        """Compute all metrics"""
        metrics = {}

        # Convert to numpy arrays
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)

        # Basic metrics
        metrics["accuracy"] = (y_true == y_pred).mean()

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        # AUC-ROC (one-vs-rest)
        if self.num_classes > 2:
            # Convert to one-hot for multiclass
            y_true_onehot = np.eye(self.num_classes)[y_true]
            metrics["auc_roc"] = roc_auc_score(y_true_onehot, y_prob, multi_class="ovr")
        else:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        for i in range(self.num_classes):
            metrics[f"precision_class_{i}"] = precision[i]
            metrics[f"recall_class_{i}"] = recall[i]
            metrics[f"f1_class_{i}"] = f1[i]

        return metrics

"""Loss functions module for handling class imbalance in skin lesion classification.

This module provides specialized loss functions designed to address the severe
class imbalance in the HAM10000 dataset (up to 58:1 ratio between classes).
These loss functions improve model training by giving appropriate weight to
minority classes and preventing overfitting.

Key loss functions:
    - FocalLoss: Addresses class imbalance by down-weighting easy examples
    - LabelSmoothingLoss: Improves generalization by preventing overconfidence
    - CrossEntropyWithSmoothing: Combines standard CE with label smoothing

The choice of loss function significantly impacts:
    1. Model convergence speed
    2. Performance on minority classes
    3. Generalization to unseen data
    4. Calibration of prediction probabilities

Typical usage:
    # For severe class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # For better generalization
    criterion = LabelSmoothingLoss(num_classes=7, smoothing=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for addressing extreme class imbalance.
    
    Focal loss modifies cross-entropy loss to down-weight well-classified
    examples, focusing learning on hard, misclassified examples. This is
    particularly effective for the HAM10000 dataset where some classes have
    50x fewer samples than others.
    
    The loss is formulated as:
        FL(pt) = -α(1-pt)^γ * log(pt)
    
    Where:
        - pt is the model's estimated probability for the true class
        - α is the weighting factor (typically 1)
        - γ is the focusing parameter (typically 2)
    
    Benefits:
        - Automatically handles class imbalance without explicit weights
        - Focuses on hard examples near decision boundary
        - Reduces relative loss for well-classified examples
        - Prevents easy negative class from dominating gradient
    
    Args:
        alpha (float): Weighting factor in range [0,1] to balance importance
            Default: 1 (no weighting)
        gamma (float): Focusing parameter for modulating loss
            gamma=0 reduces to CE loss
            gamma>0 reduces relative loss for well-classified examples
            Default: 2 (commonly used value)
        reduction (str): Specifies reduction to apply to output
            'none': no reduction
            'mean': mean of output
            'sum': sum of output
            Default: 'mean'
    
    Example:
        >>> criterion = FocalLoss(alpha=1, gamma=2)
        >>> logits = model(images)  # (batch_size, num_classes)
        >>> loss = criterion(logits, targets)  # scalar if reduction='mean'
    """

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Computed focal loss (scalar if reduction is 'mean' or 'sum')
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for improved generalization.
    
    Label smoothing prevents the model from becoming overconfident by
    replacing hard targets (0 or 1) with soft targets. Instead of
    requiring probability 1 for correct class, it requires 1-ε and
    distributes ε among other classes.
    
    This technique:
        - Prevents overfitting by penalizing overconfident predictions
        - Improves model calibration (predicted probabilities match accuracy)
        - Encourages the model to be less certain about predictions
        - Particularly helpful when training data has label noise
    
    The smoothed target distribution becomes:
        - True class: 1 - smoothing
        - Other classes: smoothing / (num_classes - 1)
    
    Args:
        num_classes (int): Number of classes in classification task
        smoothing (float): Smoothing parameter in range [0, 1]
            0: No smoothing (standard cross-entropy)
            0.1: Common choice - 90% confidence on true class
            0.2: Aggressive smoothing for noisy labels
    
    Example:
        >>> criterion = LabelSmoothingLoss(num_classes=7, smoothing=0.1)
        >>> logits = model(images)  # (batch_size, 7)
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        """Compute label smoothing loss.
        
        Args:
            pred: Predicted logits of shape (batch_size, num_classes)
            target: Ground truth labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # Create smoothed target distribution
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

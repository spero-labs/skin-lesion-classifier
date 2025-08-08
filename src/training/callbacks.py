import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric optimization
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric
            if self.verbose:
                logger.info(f"EarlyStopping: Initial best score = {self.best_score:.4f}")
        elif self.is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: New best score = {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: No improvement for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: Stopping training after {self.patience} epochs without improvement")
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_auc",
        mode: str = "max",
        save_best_only: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' for metric optimization
            save_best_only: Whether to save only best model
            save_last: Whether to always save last model
            verbose: Whether to print messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_score = None
        
        if mode == "min":
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / "checkpoint_last.pth"
            torch.save(checkpoint, last_path)
            if self.verbose:
                logger.info(f"Saved last checkpoint to {last_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            if self.verbose:
                logger.info(f"Saved best checkpoint to {best_path}")
            
            # Also save metrics
            metrics_path = self.save_dir / "best_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        
        # Save epoch checkpoint if not save_best_only
        if not self.save_best_only:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)
            if self.verbose:
                logger.info(f"Saved checkpoint to {epoch_path}")
    
    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Check and save checkpoint if needed.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Current metrics
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        is_best = False
        if self.best_score is None:
            self.best_score = current_score
            is_best = True
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            is_best = True
        
        if is_best or not self.save_best_only:
            self.save_checkpoint(model, optimizer, epoch, metrics, is_best)


class LearningRateScheduler:
    """Learning rate scheduler callback."""
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric_based: bool = False,
        verbose: bool = True
    ):
        """
        Initialize LR scheduler callback.
        
        Args:
            scheduler: PyTorch scheduler instance
            metric_based: Whether scheduler needs metric input
            verbose: Whether to print messages
        """
        self.scheduler = scheduler
        self.metric_based = metric_based
        self.verbose = verbose
    
    def __call__(self, metric: Optional[float] = None):
        """
        Step the scheduler.
        
        Args:
            metric: Optional metric for ReduceLROnPlateau
        """
        if self.metric_based:
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        if self.verbose:
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Learning rate: {current_lr:.2e}")


class GradientAccumulator:
    """Gradient accumulation for larger effective batch sizes."""
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def get_scale_factor(self) -> float:
        """Get loss scaling factor."""
        return 1.0 / self.accumulation_steps


class MetricTracker:
    """Track and log metrics during training."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize metric tracker.
        
        Args:
            save_dir: Optional directory to save metrics
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
    
    def update(self, metrics: Dict[str, float], phase: str = "train"):
        """
        Update metrics for a phase.
        
        Args:
            metrics: Metrics dictionary
            phase: One of 'train', 'val', 'test'
        """
        if phase == "train":
            self.train_metrics.append(metrics)
        elif phase == "val":
            self.val_metrics.append(metrics)
        elif phase == "test":
            self.test_metrics.append(metrics)
    
    def save(self):
        """Save metrics to JSON file."""
        if self.save_dir is None:
            return
        
        metrics_data = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        
        metrics_path = self.save_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
    
    def get_best_metric(self, metric_name: str, phase: str = "val", mode: str = "max") -> float:
        """
        Get best value of a metric.
        
        Args:
            metric_name: Name of the metric
            phase: Phase to check
            mode: 'min' or 'max'
        
        Returns:
            Best metric value
        """
        if phase == "train":
            metrics_list = self.train_metrics
        elif phase == "val":
            metrics_list = self.val_metrics
        else:
            metrics_list = self.test_metrics
        
        if not metrics_list:
            return None
        
        values = [m.get(metric_name) for m in metrics_list if metric_name in m]
        
        if not values:
            return None
        
        if mode == "max":
            return max(values)
        else:
            return min(values)
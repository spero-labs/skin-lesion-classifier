import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Any
import logging
from pathlib import Path

from ..models.base_model import BaseModel
from ..utils.config import TrainingConfig
from .losses import FocalLoss, LabelSmoothingLoss
from .metrics import MetricsCalculator
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    GradientAccumulator,
    MetricTracker,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Main training class for skin lesion classification."""

    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        device: str = None,
        use_wandb: bool = True,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use (cuda/cpu)
            use_wandb: Whether to use Weights & Biases
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.config = config
        
        # Better device selection
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Mixed precision training (only for CUDA)
        self.scaler = GradScaler() if self.device == "cuda" else None
        
        # Setup callbacks
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode="max",
            verbose=True
        )
        
        self.checkpoint = ModelCheckpoint(
            save_dir=str(self.checkpoint_dir),
            monitor="auc_macro",
            mode="max",
            save_best_only=False,
            save_last=True
        )
        
        self.lr_scheduler = LearningRateScheduler(
            self.scheduler,
            metric_based=isinstance(self.scheduler, ReduceLROnPlateau),
            verbose=True
        )
        
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=config.gradient_accumulation_steps
            if hasattr(config, 'gradient_accumulation_steps') else 1
        )
        
        self.metric_tracker = MetricTracker(save_dir=str(self.checkpoint_dir))
        
        # Metrics calculator
        self.train_metrics = MetricsCalculator(
            num_classes=model.num_classes,
            class_names=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        )
        self.val_metrics = MetricsCalculator(
            num_classes=model.num_classes,
            class_names=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        )
        
        # Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb()
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0

    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        optimizer_name = getattr(self.config, 'optimizer', 'adamw')
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        
        # Get parameter groups for differential learning rates
        param_groups = self.model.get_param_groups(lr, lr_backbone=lr/10)
        
        if optimizer_name.lower() == 'adam':
            return Adam(param_groups, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return SGD(
                param_groups,
                momentum=getattr(self.config, 'momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            return AdamW(param_groups, weight_decay=weight_decay)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_name = self.config.scheduler
        
        if scheduler_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif scheduler_name == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name == "step":
            return StepLR(
                self.optimizer,
                step_size=getattr(self.config, 'step_size', 10),
                gamma=getattr(self.config, 'gamma', 0.1)
            )
        else:
            return None
    
    def _setup_loss(self):
        """Setup loss function."""
        loss_name = getattr(self.config, 'loss', 'focal')
        
        if loss_name == "focal":
            return FocalLoss(alpha=1, gamma=2)
        elif loss_name == "label_smoothing":
            return LabelSmoothingLoss(
                num_classes=self.model.num_classes,
                smoothing=getattr(self.config, 'label_smoothing', 0.1)
            )
        elif loss_name == "ce":
            return nn.CrossEntropyLoss()
        else:
            return FocalLoss(alpha=1, gamma=2)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=getattr(self.config, 'project_name', 'skin-lesion-classification'),
                name=getattr(self.config, 'experiment_name', 'baseline'),
                config=self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
            )
            wandb.watch(self.model, log="all", log_freq=100)
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 3:
                images, targets, metadata = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                metadata = metadata.to(self.device) if metadata is not None else None
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                metadata = None
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    if hasattr(self.model, 'use_metadata') and self.model.use_metadata and metadata is not None:
                        outputs = self.model(images, metadata)
                    else:
                        outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss * self.gradient_accumulator.get_scale_factor()
            else:
                if hasattr(self.model, 'use_metadata') and self.model.use_metadata and metadata is not None:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss * self.gradient_accumulator.get_scale_factor()
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if self.gradient_accumulator.should_step():
                if self.scaler is not None:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            running_loss += loss.item() / self.gradient_accumulator.get_scale_factor()
            probs = F.softmax(outputs.detach(), dim=1)
            preds = outputs.detach().argmax(1)
            
            self.train_metrics.update(preds, targets, probs)
            
            # Update progress bar
            pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics["loss"] = running_loss / len(dataloader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            # Handle different batch formats
            if len(batch) == 3:
                images, targets, metadata = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                metadata = metadata.to(self.device) if metadata is not None else None
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                metadata = None
            
            # Forward pass
            if hasattr(self.model, 'use_metadata') and self.model.use_metadata and metadata is not None:
                outputs = self.model(images, metadata)
            else:
                outputs = self.model(images)
            
            loss = self.criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            self.val_metrics.update(preds, targets, probs)
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute()
        metrics["loss"] = running_loss / len(dataloader)
        
        return metrics
    
    def fit(self, train_dataloader, val_dataloader):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            self.metric_tracker.update(train_metrics, "train")
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            self.metric_tracker.update(val_metrics, "val")
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"AUC: {train_metrics.get('auc_macro', 0):.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"AUC: {val_metrics.get('auc_macro', 0):.4f}")
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/auc": train_metrics.get("auc_macro", 0),
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/auc": val_metrics.get("auc_macro", 0),
                })
            
            # Callbacks
            self.checkpoint(self.model, self.optimizer, epoch, val_metrics)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.lr_scheduler(val_metrics.get("auc_macro", val_metrics["accuracy"]))
            else:
                self.lr_scheduler()
            
            # Early stopping
            if self.early_stopping(val_metrics.get("auc_macro", val_metrics["accuracy"])):
                logger.info("Early stopping triggered!")
                break
            
            # Update best score
            current_auc = val_metrics.get("auc_macro", 0)
            if current_auc > self.best_val_auc:
                self.best_val_auc = current_auc
                logger.info(f"New best validation AUC: {self.best_val_auc:.4f}")
        
        # Save final metrics
        self.metric_tracker.save()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info(f"Training completed! Best validation AUC: {self.best_val_auc:.4f}")
        
        return self.best_val_auc
    
    @torch.no_grad()
    def test(self, test_dataloader):
        """Test the model."""
        logger.info("Running test evaluation...")
        
        test_metrics = self.validate(test_dataloader)
        self.metric_tracker.update(test_metrics, "test")
        
        # Print classification report
        logger.info("\nTest Classification Report:")
        logger.info(self.val_metrics.get_classification_report())
        
        # Log test metrics
        logger.info(f"\nTest Metrics:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  AUC-ROC: {test_metrics.get('auc_macro', 0):.4f}")
        logger.info(f"  Sensitivity: {test_metrics['avg_sensitivity']:.4f}")
        logger.info(f"  Specificity: {test_metrics['avg_specificity']:.4f}")
        
        return test_metrics
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
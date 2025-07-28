import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from typing import Dict, Optional


class Trainer:
    """Main training class"""

    def __init__(self, model: BaseModel, config: TrainingConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler()  # For mixed precision

        # Setup losses
        self.criterion = FocalLoss()

        # Metrics tracking
        self.best_val_auc = 0
        self.patience_counter = 0

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        predictions = []
        labels = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Mixed precision training
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_val
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            running_loss += loss.item()
            predictions.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})

        return running_loss / len(dataloader), predictions, labels

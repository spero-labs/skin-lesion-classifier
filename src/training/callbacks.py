from pathlib import Path
import torch
from typing import Dict


class ModelCheckpoint:
    """Save model checkpoints"""

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_auc",
        mode: str = "max",
        save_top_k: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_scores = []

    def __call__(self, epoch: int, model: torch.nn.Module, metrics: Dict):
        score = metrics[self.monitor]

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "score": score,
        }

        # Save latest
        torch.save(checkpoint, self.save_dir / "latest.pth")

        # Save best
        if self._is_better(score):
            torch.save(checkpoint, self.save_dir / f"best_{self.monitor}.pth")

        # Save top-k
        self._save_top_k(epoch, checkpoint, score)


class EarlyStopping:
    """Early stopping callback"""

    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_better(score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

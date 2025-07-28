class TestTimeAugmentation:
    """Apply augmentations at test time"""

    def __init__(self, transforms: List, n_augmentations: int = 5):
        self.transforms = transforms
        self.n_augmentations = n_augmentations

    def predict(self, model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
        """Average predictions over augmented versions"""
        predictions = []

        for _ in range(self.n_augmentations):
            augmented = self.apply_transform(image)
            with torch.no_grad():
                pred = model(augmented)
                predictions.append(pred)

        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)

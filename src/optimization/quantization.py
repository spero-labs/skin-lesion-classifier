import torch.quantization as quantization


def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize model for faster inference"""
    model.eval()
    model.qconfig = quantization.get_default_qconfig("fbgemm")
    quantization.prepare(model, inplace=True)
    quantization.convert(model, inplace=True)
    return model


class ModelEnsemble:
    """Ensemble multiple models"""

    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        self.models = [self.load_model(path) for path in model_paths]
        self.weights = weights or [1.0 / len(self.models)] * len(self.models)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model(x) * weight
            predictions.append(pred)
        return torch.sum(torch.stack(predictions), dim=0)


class MCDropout:
    """Monte Carlo Dropout for uncertainty estimation"""

    def __init__(self, model: torch.nn.Module, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples

    def predict_with_uncertainty(self, x: torch.Tensor):
        """Get prediction with uncertainty"""
        self.enable_dropout()

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std

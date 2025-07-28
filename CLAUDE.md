# CLAUDE.md

## Project Overview
Build a production-ready skin lesion classification system that can distinguish between different types of skin lesions (benign vs malignant, or multi-class classification). The system should be modular, maintainable, and achieve strong performance metrics.

## Core Requirements
1. Data Handling Module
Requirements:
- Implement a flexible DataLoader class that can handle multiple datasets (HAM10000, ISIC, etc.)
- Support for train/validation/test splits with stratification
- Implement proper data augmentation pipeline for medical images
- Handle class imbalance through weighted sampling or oversampling
- Cache processed images for faster training

2. Model Architecture Module
Requirements:
- Implement at least 3 different architectures:
    - EfficientNet-B0/B1 (lightweight, good for deployment)
    - ResNet50/101 with attention mechanisms
    - Vision Transformer (ViT) or Swin Transformer
- Use transfer learning with ImageNet pre-trained weights
- Add custom head with dropout and batch normalization
- Implement model ensembling capability
Advanced Features:
- Global Average Pooling + Attention mechanism
- Multi-scale feature extraction
- Auxiliary metadata input (age, sex, location) fusion
- Uncertainty estimation (MC Dropout or ensemble)

3. Training Pipeline
Requirements:
- Implement a comprehensive Trainer class with:
    - Mixed precision training (FP16)
    - Gradient accumulation for larger effective batch sizes
    - Learning rate scheduling (Cosine annealing with warm restarts)
    - Early stopping with patience
    - Model checkpointing (save best and last)

4. Evaluation & Metrics
Implement comprehensive evaluation:
    - Primary metrics: AUC-ROC, Sensitivity, Specificity
    - Secondary metrics: F1-score, Precision, Recall per class
    - Confusion matrix visualization
    - ROC curves for each class
    - Grade CAM/Attention visualization for interpretability
    - Statistical significance testing between models

5. Configuration Management
Use YAML config files for all hyperparameters:
```yaml
data:
  dataset: "HAM10000"
  image_size: 224
  batch_size: 32
  num_workers: 4
  augmentation:
    rotation: 30
    zoom: 0.2
    horizontal_flip: true
    color_jitter: 0.2

model:
  architecture: "efficientnet_b1"
  pretrained: true
  dropout: 0.3
  num_classes: 7

training:
  epochs: 50
  learning_rate: 1e-3
  weight_decay: 1e-4
  scheduler: "cosine"
  early_stopping_patience: 10
```

6. Performance Optimization
Requirements:
    Achieve >85% balanced accuracy on validation set
    Implement test-time augmentation (TTA)
    Use ensemble of top 3 models
    Optimize for both accuracy and inference speed
    Implement knowledge distillation for model compression

7. Inference Module
Create a clean inference API:
```python
class SkinLesionPredictor:
    def __init__(self, model_path, config_path):
        # Load model and preprocessing pipeline
    
    def predict(self, image_path, return_probabilities=True):
        # Single image prediction with confidence scores
    
    def predict_batch(self, image_paths):
        # Efficient batch prediction
    
    def explain_prediction(self, image_path):
        # Return prediction with GradCAM visualization
```

8. Code Quality Requirements
Type hints for all functions
Comprehensive docstrings (Google style)
Unit tests for data loading and preprocessing
Integration tests for training pipeline
Code should pass flake8 and black formatting
Implement proper logging throughout
Error handling for edge cases

9. Additional Features
Web API: Create FastAPI endpoint for model serving
Docker: Containerize the application
Model versioning: Track experiments with MLflow/W&B
Data versioning: Use DVC for dataset management
CI/CD: GitHub Actions for testing and deployment

10. Documentation
Include:
    README with setup instructions
    Model card with performance metrics
    API documentation
    Jupyter notebook with EDA and results visualization
    Training reproduction instructions

Performance Targets
    Validation AUC-ROC: >0.90
    Inference time: <100ms per image
    Model size: <50MB (quantized)
    Memory usage: <2GB during inference

## Common Commands

### Running the Training Pipeline
```bash
make train
```
The training uses Hydra configuration framework. Configuration files are in `configs/` directory.

### Running Tests
```bash
make test
```

### Code Formatting and Linting
```bash
make format  # Format code with black
make lint    # Check code style with flake8
```

### Other Useful Commands
```bash
make install  # Install dependencies
make serve    # Start FastAPI server
make clean    # Clean cache and temporary files
make help     # Show all available commands
```

## Project Architecture

### Key Components

1. **Data Pipeline** (`src/data/`)
   - `dataset.py`: Handles loading of HAM10000 skin lesion dataset
   - `preprocessing.py`: Image preprocessing utilities
   - `augmentation.py`: Data augmentation pipelines using albumentations
   - `dataloader.py`: PyTorch DataLoader configurations

2. **Model Architecture** (`src/models/`)
   - `base_model.py`: Base model classes
   - `architectures.py`: Model architectures (likely using timm for transfer learning)
   - `model_factory.py`: Factory pattern for model creation

3. **Training Pipeline** (`src/training/`)
   - `trainer.py`: Main training loop implementation
   - `losses.py`: Custom loss functions
   - `metrics.py`: Evaluation metrics
   - `callbacks.py`: Training callbacks (checkpointing, early stopping, etc.)

4. **Inference** (`src/inference/`)
   - `predictor.py`: Inference pipeline for predictions
   - `tta.py`: Test-time augmentation implementation

5. **Utilities** (`src/utils/`)
   - `config.py`: Configuration management
   - `logger.py`: Logging utilities
   - `visualization.py`: Visualization tools for results
   - `interpretability.py`: Model interpretability (likely Grad-CAM)

6. **API** (`src/api/`)
   - `app.py`: FastAPI application for model serving

7. **Optimization** (`src/optimization/`)
   - `quantization.py`: Model quantization for deployment

### Dataset
The project uses the HAM10000 dataset (Human Against Machine with 10000 training images) for skin lesion classification. The dataset contains images in `HAM10000/` directory with associated metadata CSV files.

### Dependencies
- PyTorch and PyTorch Lightning for deep learning
- timm for pre-trained models
- albumentations for data augmentation
- Hydra for configuration management
- wandb for experiment tracking
- FastAPI for model serving
- grad-cam for model interpretability

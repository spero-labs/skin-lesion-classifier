# Project Structure Guide

## Overview
The Skin Lesion Classification System is organized into modular components for maintainability and scalability.

## Directory Structure

```
skin-lesion-classifier/
│
├── src/                      # Main source code
│   ├── data/                    # Data pipeline
│   │   ├── dataset.py          # PyTorch dataset implementation
│   │   ├── dataloader.py       # DataLoader with sampling strategies
│   │   ├── preprocessing.py    # Image preprocessing utilities
│   │   └── augmentation.py     # Data augmentation pipelines
│   │
│   ├── models/                  # Model architectures
│   │   ├── base_model.py       # Abstract base model class
│   │   ├── architectures.py    # EfficientNet, ResNet, ViT, Swin
│   │   ├── model_factory.py    # Factory pattern for model creation
│   │   └── ensemble.py         # Ensemble model implementation
│   │
│   ├── training/                # Training pipeline
│   │   ├── trainer.py          # Main training loop
│   │   ├── metrics.py          # Metrics calculation
│   │   ├── callbacks.py        # Training callbacks
│   │   └── losses.py           # Custom loss functions
│   │
│   ├── inference/               # Inference pipeline
│   │   ├── predictor.py        # Prediction interface
│   │   └── tta.py              # Test-time augmentation
│   │
│   ├── api/                     # API implementation
│   │   └── app.py              # FastAPI application
│   │
│   ├── optimization/            # Model optimization
│   │   └── quantization.py     # Model quantization utilities
│   │
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration management
│       ├── logger.py           # Logging setup
│       ├── visualization.py    # Visualization tools
│       └── interpretability.py # Model interpretability (Grad-CAM)
│
├── scripts/                  # Organized utility scripts
│   ├── training/               # Training-related scripts
│   │   ├── train.py           # Main training script
│   │   ├── quick_train.sh     # Quick training (10 epochs)
│   │   └── test_system.py     # System component testing
│   │
│   ├── analysis/               # Analysis and visualization
│   │   ├── analyze_results.py          # Results analysis
│   │   ├── generate_visualizations.py  # Generate plots
│   │   └── save_model_from_training.py # Checkpoint utilities
│   │
│   ├── system/                 # System configuration
│   │   └── fix_macos_limits.sh # Fix macOS file limits
│   │
│   └── deployment/             # Deployment utilities
│       └── main.py            # Application entry point
│
├── configs/                  # Configuration files
│   └── config.yaml             # Hydra configuration
│
├── tests/                    # Unit and integration tests
│   ├── test_data/              # Data pipeline tests
│   ├── test_models/            # Model tests
│   └── test_training/          # Training tests
│
├── notebooks/                # Jupyter notebooks
│   ├── EDA.ipynb               # Exploratory data analysis
│   └── model_evaluation.ipynb  # Model evaluation
│
├── checkpoints/              # Saved models
│   ├── checkpoint_best.pth     # Best model weights
│   ├── checkpoint_last.pth     # Latest checkpoint
│   └── training_metrics.json   # Training history
│
├── visualizations/           # Generated plots
│   ├── training_curves.png     # Training/validation curves
│   ├── performance_analysis.png # Performance metrics
│   └── class_imbalance_analysis.png # Class distribution
│
├── HAM10000/                 # Dataset
│   ├── HAM10000_images_part_1/ # Image files (part 1)
│   ├── HAM10000_images_part_2/ # Image files (part 2)
│   └── HAM10000_metadata.csv   # Metadata
│
├── outputs/                  # Hydra outputs
│   └── YYYY-MM-DD/             # Timestamped runs
│
├── Makefile                  # Command interface
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── CLAUDE.md                 # Development guidelines
└── .gitignore               # Git ignore rules
```

## Key Components

### 1. Data Pipeline (`src/data/`)
- **dataset.py**: Handles HAM10000 dataset loading with stratified splits
- **dataloader.py**: Implements weighted sampling for class imbalance
- **augmentation.py**: Comprehensive augmentation using albumentations
- **preprocessing.py**: Image normalization and resizing

### 2. Model Architecture (`src/models/`)
- **EfficientNet**: Lightweight, efficient models (B0-B3)
- **ResNet with Attention**: Enhanced feature extraction
- **Vision Transformer**: State-of-the-art transformer models
- **Swin Transformer**: Hierarchical vision transformers
- **Ensemble**: Combines multiple models for better performance

### 3. Training Pipeline (`src/training/`)
- **Mixed Precision**: FP16 training for efficiency
- **Gradient Accumulation**: Larger effective batch sizes
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Checkpointing**: Saves best and latest models

### 4. Inference (`src/inference/`)
- **Test-Time Augmentation**: Improved predictions
- **Batch Processing**: Efficient multi-image inference
- **Confidence Scores**: Probability outputs for all classes

### 5. Scripts Organization (`scripts/`)
All executable scripts are organized by purpose:
- **training/**: Scripts for model training
- **analysis/**: Results analysis and visualization
- **system/**: System configuration utilities
- **deployment/**: Production deployment scripts

## Configuration Management

### Hydra Configuration (`configs/config.yaml`)
```yaml
model:
  architecture: efficientnet_b0
  num_classes: 7
  dropout: 0.3

data:
  batch_size: 32
  image_size: 224
  num_workers: 0

training:
  epochs: 50
  learning_rate: 0.001
  scheduler: plateau
```

### Override Examples
```bash
# Change model architecture
python scripts/training/train.py model.architecture=resnet50

# Adjust training parameters
python scripts/training/train.py training.epochs=100 training.learning_rate=0.0001

# Use different device
python scripts/training/train.py experiment.device=cuda
```

## Development Workflow

### 1. Setup Environment
```bash
make install          # Install dependencies
make install-dev      # Install dev tools
make fix-limits       # Fix macOS limits
```

### 2. Explore Data
```bash
make dataset-info     # View dataset statistics
jupyter notebook      # Open notebooks for EDA
```

### 3. Train Model
```bash
make train            # Full training
make quick-train      # Quick test run
make train-custom     # Interactive configuration
```

### 4. Analyze Results
```bash
make analyze          # Analyze metrics
make visualize        # Generate plots
make show-results     # Quick summary
```

### 5. Development
```bash
make format           # Format code
make lint            # Check style
make test            # Run tests
make pre-commit      # All checks
```

### 6. Deployment
```bash
make serve           # Start API server
# Note: Docker support removed for simplicity
# Add containerization when needed for production
```

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Notebooks**: `PascalCase.ipynb`
- **Configs**: `lowercase.yaml`
- **Scripts**: `snake_case.py` or `kebab-case.sh`
- **Checkpoints**: `checkpoint_*.pth`
- **Logs**: `YYYY-MM-DD_HH-MM-SS.log`

## Best Practices

### Code Organization
1. Each module should have a single responsibility
2. Use abstract base classes for extensibility
3. Implement factory patterns for object creation
4. Keep configuration separate from code

### Data Management
1. Use stratified splits for train/val/test
2. Implement proper data augmentation
3. Handle class imbalance with sampling
4. Cache processed data when possible

### Training
1. Always use validation set for model selection
2. Implement early stopping to prevent overfitting
3. Save checkpoints regularly
4. Log all metrics for analysis

### Testing
1. Write unit tests for data pipeline
2. Test model forward pass
3. Validate metrics calculation
4. Integration tests for full pipeline

## Common Tasks

### Add New Model Architecture
1. Create class in `src/models/architectures.py`
2. Register in `src/models/model_factory.py`
3. Update config options in `configs/config.yaml`

### Add New Dataset
1. Create dataset class in `src/data/`
2. Update dataloader in `src/data/dataloader.py`
3. Add configuration options

### Add New Loss Function
1. Implement in `src/training/losses.py`
2. Register in trainer configuration
3. Update config options

### Add New Metric
1. Implement in `src/training/metrics.py`
2. Add to MetricsCalculator class
3. Update logging and visualization

## Troubleshooting

### Import Errors
- Ensure project root is in PYTHONPATH
- Use absolute imports: `from src.models import ...`

### Memory Issues
- Reduce batch size in config
- Enable gradient accumulation
- Use mixed precision training

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Optimize data loading with workers
- Use smaller image size or model

## Contributing

1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Run `make pre-commit` before pushing
5. Create descriptive commit messages

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hydra Configuration](https://hydra.cc/)
- [timm Models](https://github.com/rwightman/pytorch-image-models)
- [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
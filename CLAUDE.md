# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
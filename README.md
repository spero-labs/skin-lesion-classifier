# Skin Lesion Classification System

A production-ready deep learning system for classifying skin lesions using the HAM10000 dataset. The system can distinguish between 7 types of skin lesions with high accuracy using state-of-the-art architectures and training techniques.

## Features

- **Multiple Model Architectures**: EfficientNet, ResNet with Attention, Vision Transformer, Swin Transformer
- **Advanced Training**: Mixed precision (FP16), gradient accumulation, early stopping, learning rate scheduling
- **Class Imbalance Handling**: Weighted sampling, focal loss, class-balanced metrics
- **Comprehensive Metrics**: AUC-ROC, sensitivity, specificity, F1-score per class
- **Production Ready**: Model checkpointing, experiment tracking, test-time augmentation
- **Modular Design**: Clean separation of data, models, training, and inference components

## Dataset

The system uses the HAM10000 dataset with 10,015 dermoscopic images across 7 classes:
- **akiec**: Actinic keratoses and intraepithelial carcinoma (327 images)
- **bcc**: Basal cell carcinoma (514 images)
- **bkl**: Benign keratosis-like lesions (1,099 images)
- **df**: Dermatofibroma (115 images)
- **mel**: Melanoma (1,113 images)
- **nv**: Melanocytic nevi (6,705 images)
- **vasc**: Vascular lesions (142 images)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd skin-lesion-classifier

# Install dependencies
pip install -r requirements.txt
# or
make install
```

## Quick Start

1. **Test the system**:
```bash
python test_system.py
```

2. **Train a model**:
```bash
# Using default configuration
make train

# Or with custom parameters
python train.py model.architecture=efficientnet_b1 training.epochs=100 data.batch_size=16
```

3. **Train with different models**:
```bash
# EfficientNet-B0 (lightweight, fast)
python train.py model.architecture=efficientnet_b0

# ResNet50 with attention
python train.py model.architecture=resnet50

# Vision Transformer
python train.py model.architecture=vit_small

# Ensemble of models
python train.py advanced.use_ensemble=true
```

## Configuration

The system uses Hydra for configuration management. Main configuration file: `configs/config.yaml`

Key configuration options:
- `data.batch_size`: Batch size for training (default: 32)
- `data.image_size`: Input image size (default: 224)
- `model.architecture`: Model architecture to use
- `training.epochs`: Number of training epochs (default: 50)
- `training.learning_rate`: Learning rate (default: 0.001)
- `experiment.use_wandb`: Enable Weights & Biases logging

## Project Structure

```
skin-lesion-classifier/
├── configs/               # Configuration files
│   └── config.yaml       # Main configuration
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architectures
│   ├── training/        # Training logic and metrics
│   ├── inference/       # Inference and prediction
│   └── utils/           # Utilities
├── HAM10000/            # Dataset directory
├── train.py             # Main training script
├── test_system.py       # System test script
└── Makefile            # Common commands
```

## Available Commands

```bash
make help        # Show all available commands
make train       # Train the model
make test        # Run tests
make format      # Format code with black
make lint        # Check code style with flake8
make clean       # Clean cache files
```

## Model Performance

Target metrics:
- Validation AUC-ROC: >0.90
- Balanced Accuracy: >0.85
- Inference time: <100ms per image

## Training Tips

1. **For quick experiments**:
```bash
python train.py training.epochs=10 data.batch_size=16
```

2. **For best performance**:
```bash
python train.py model.architecture=efficientnet_b1 training.epochs=100 \
    training.learning_rate=0.0001 training.scheduler=cosine
```

3. **For handling class imbalance**:
```bash
python train.py training.loss=focal data.use_weighted_sampling=true
```

4. **For using metadata** (age, sex, location):
```bash
python train.py model.use_metadata=true data.use_metadata=true
```

## Monitoring Training

The system logs training progress to:
- Console output with progress bars
- `training.log` file
- Weights & Biases (if enabled with `experiment.use_wandb=true`)
- Checkpoint files in `checkpoints/` directory

## Inference

After training, use the saved model for predictions:

```python
from src.inference import SkinLesionPredictor

# Load trained model
predictor = SkinLesionPredictor(
    model_path="checkpoints/checkpoint_best.pth",
    use_tta=True  # Enable test-time augmentation
)

# Make prediction
result = predictor.predict("path/to/image.jpg")
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Troubleshooting

### macOS Specific Issues

1. **"Too many open files" error**: 
```bash
# Quick fix for current session
ulimit -n 4096

# Or use the provided script
./fix_macos_limits.sh

# Then run training with num_workers=0
python train.py data.num_workers=0
```

2. **MPS (Metal) device issues**:
```bash
# Use CPU if MPS causes problems
python train.py experiment.device=cpu

# Or reduce batch size for MPS
python train.py data.batch_size=8 experiment.device=mps
```

### General Issues

1. **CUDA out of memory**: Reduce batch size
```bash
python train.py data.batch_size=8
```

2. **Slow training**: Reduce image size or use smaller model
```bash
python train.py data.image_size=192 model.architecture=efficientnet_b0
```

3. **Poor performance**: Try different loss or more epochs
```bash
python train.py training.loss=focal training.epochs=100
```

## Quick Training (macOS Optimized)

For a quick training session optimized for macOS:
```bash
./quick_train.sh
```

This script automatically:
- Sets file descriptor limits
- Uses appropriate batch size
- Disables multiprocessing (num_workers=0)
- Uses MPS device for M1/M2 Macs

## License

This project is for educational and research purposes.

## Acknowledgments

- HAM10000 dataset: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Model architectures from [timm](https://github.com/rwightman/pytorch-image-models)
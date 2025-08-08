# Changelog

## [1.0.1] - 2024-08-07

### Changed
- Removed Docker support for simplicity
  - Deleted Dockerfile 
  - Removed Docker commands from Makefile
  - Project focuses on local development and training
  - Docker can be added later when deployment is needed

### Rationale
- FastAPI app is incomplete
- Current focus is on model training and analysis
- Local setup works well without containerization
- Reduces complexity for research/development phase

## [1.0.0] - 2024-08-07

### Added
- Complete skin lesion classification system implementation
- Support for 7 classes of skin lesions from HAM10000 dataset
- Multiple model architectures (EfficientNet, ResNet, ViT, Swin Transformer)
- Comprehensive training pipeline with mixed precision and gradient accumulation
- Test-time augmentation for improved inference
- Extensive visualization and analysis tools
- 40+ Makefile commands for all operations
- Organized project structure with scripts directory

### Project Organization
- **Scripts organized into directories**:
  - `scripts/training/` - Training-related scripts
  - `scripts/analysis/` - Analysis and visualization tools
  - `scripts/system/` - System configuration utilities
  - `scripts/deployment/` - Deployment scripts
- **No symlinks** - Clean directory structure without symlinks
- **Comprehensive Makefile** with categorized commands
- **Full documentation** in README and PROJECT_STRUCTURE.md

### Performance
- Achieved 97.58% AUC-ROC on test set
- 87.49% accuracy across 7 classes
- 82.67% balanced accuracy
- Excellent melanoma detection: 82% sensitivity, 90.8% specificity

### Features
- **Data Pipeline**:
  - Stratified train/val/test splits
  - Weighted sampling for class imbalance
  - Comprehensive augmentation pipeline
  
- **Training**:
  - Mixed precision training (FP16)
  - Gradient accumulation
  - Early stopping with patience
  - Learning rate scheduling
  - Model checkpointing
  
- **Analysis**:
  - Training curves visualization
  - Per-class performance analysis
  - Class imbalance impact analysis
  - Comprehensive metrics tracking

### Technical Details
- PyTorch-based implementation
- Hydra configuration management
- timm library for pretrained models
- Albumentations for data augmentation
- Support for MPS (Mac M1/M2) and CUDA devices

### Documentation
- Comprehensive README with visualizations
- PROJECT_STRUCTURE.md with detailed guide
- Inline documentation and type hints
- Analysis reports with clinical significance

### Known Issues
- Model checkpoint saving was fixed after initial training
- macOS file descriptor limits require adjustment (solution provided)

### Future Improvements
- Complete FastAPI implementation
- Add Grad-CAM visualization
- Implement model quantization
- Add external validation on ISIC datasets

## Installation

```bash
# Clone and setup
git clone <repository>
cd skin-lesion-classifier
make install
make fix-limits  # For macOS

# Train model
make train

# Analyze results  
make analyze
make visualize
```

## Commands Reference

Run `make help` to see all available commands organized by category:
- Setup & Installation
- Training & Testing
- Analysis & Visualization
- Development Tools
- API & Serving
- Docker Support
- Jupyter Integration
- Cleanup Options
- Information Commands

## Contributors
- Model implementation and training pipeline
- Data processing and augmentation
- Visualization and analysis tools
- Documentation and organization
# Scripts Directory

Organized collection of utility scripts for the Skin Lesion Classification System.

## Directory Structure

```
scripts/
├── training/        # Training-related scripts
│   ├── train.py            # Main training script
│   ├── quick_train.sh      # Quick training for testing
│   └── test_system.py      # System component testing
├── analysis/        # Analysis and visualization scripts
│   ├── analyze_results.py  # Analyze training results
│   ├── generate_visualizations.py  # Generate plots
│   └── save_model_from_training.py # Save checkpoint utilities
├── system/          # System configuration scripts
│   └── fix_macos_limits.sh # Fix macOS file limits
└── deployment/      # Deployment utilities
    └── main.py             # Main application entry
```

## Usage

All scripts can be run directly or through the Makefile:

### Training Scripts
```bash
# Direct execution
python scripts/training/train.py

# Via Makefile
make train
make quick-train
make test-system
```

### Analysis Scripts
```bash
# Direct execution
python scripts/analysis/analyze_results.py
python scripts/analysis/generate_visualizations.py

# Via Makefile
make analyze
make visualize
```

### System Scripts
```bash
# Direct execution
./scripts/system/fix_macos_limits.sh

# Via Makefile
make fix-limits
```

## Script Descriptions

### Training
- **train.py**: Main training script with Hydra configuration
- **quick_train.sh**: Bash script for quick 10-epoch training sessions
- **test_system.py**: Tests all system components (data, model, training)

### Analysis
- **analyze_results.py**: Analyzes saved metrics and displays summary
- **generate_visualizations.py**: Creates comprehensive training plots
- **save_model_from_training.py**: Utility to save model checkpoints

### System
- **fix_macos_limits.sh**: Fixes "too many open files" error on macOS

### Deployment
- **main.py**: Main application entry point for production deployment
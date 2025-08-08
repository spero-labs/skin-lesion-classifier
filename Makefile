.PHONY: help install format lint test train clean serve analyze visualize quick-train test-system fix-limits notebook all

# Default target
all: help

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║         Skin Lesion Classification System - Commands          ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make install         - Install all dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make fix-limits      - Fix macOS file descriptor limits"
	@echo ""
	@echo "🚀 Training & Testing:"
	@echo "  make train           - Train model with default config (50 epochs)"
	@echo "  make quick-train     - Quick training session (10 epochs)"
	@echo "  make train-custom    - Train with custom parameters (interactive)"
	@echo "  make test-system     - Test system components"
	@echo "  make test            - Run unit tests with pytest"
	@echo ""
	@echo "📊 Analysis & Visualization:"
	@echo "  make analyze         - Analyze training results"
	@echo "  make visualize       - Generate all visualizations"
	@echo "  make show-results    - Display training metrics summary"
	@echo ""
	@echo "🔧 Development Tools:"
	@echo "  make format          - Format code with black"
	@echo "  make lint            - Check code style with flake8"
	@echo "  make type-check      - Run type checking with mypy"
	@echo "  make pre-commit      - Run all checks before committing"
	@echo ""
	@echo "🌐 API & Serving:"
	@echo "  make serve           - Start FastAPI server"
	@echo "  make serve-dev       - Start server in development mode"
	@echo "  make api-docs        - Open API documentation in browser"
	@echo ""
	@echo "📓 Jupyter:"
	@echo "  make notebook        - Start Jupyter notebook server"
	@echo "  make notebook-clean  - Clean notebook outputs"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean           - Clean cache and temporary files"
	@echo "  make clean-all       - Clean everything including checkpoints"
	@echo "  make clean-logs      - Clean log files"
	@echo ""
	@echo "📋 Info:"
	@echo "  make info            - Show project information"
	@echo "  make check-gpu       - Check GPU/MPS availability"
	@echo "  make dataset-info    - Show dataset statistics"

# Setup & Installation
install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt 2>/dev/null || pip install black flake8 pytest mypy jupyter
	@echo "✅ Development dependencies installed!"

fix-limits:
	@echo "Fixing macOS file descriptor limits..."
	@./scripts/system/fix_macos_limits.sh

# Training & Testing
train:
	@echo "🚀 Starting training with default configuration..."
	python scripts/training/train.py

quick-train:
	@echo "⚡ Starting quick training session (10 epochs)..."
	@./scripts/training/quick_train.sh

train-custom:
	@echo "🎯 Custom training - Enter parameters:"
	@echo "Example: model.architecture=efficientnet_b1 training.epochs=100"
	@read -p "Parameters: " params; \
	python scripts/training/train.py $$params

test-system:
	@echo "🧪 Testing system components..."
	python scripts/training/test_system.py

test:
	@echo "🧪 Running unit tests..."
	pytest tests/ -v --color=yes || echo "No tests found. Create tests in tests/ directory."

# Analysis & Visualization
analyze:
	@echo "📊 Analyzing training results..."
	python scripts/analysis/analyze_results.py

visualize:
	@echo "📈 Generating visualizations..."
	python scripts/analysis/generate_visualizations.py

show-results:
	@echo "📋 Training Results Summary:"
	@echo "════════════════════════════════════════"
	@python -c "import json; m = json.load(open('checkpoints/training_metrics.json')); \
		val = m['val'][-1] if m.get('val') else {}; \
		print(f'AUC-ROC: {val.get(\"auc_macro\", 0):.4f}'); \
		print(f'Accuracy: {val.get(\"accuracy\", 0):.4f}'); \
		print(f'Balanced Accuracy: {val.get(\"balanced_accuracy\", 0):.4f}')" 2>/dev/null || \
		echo "No training results found. Run 'make train' first."

# Development Tools
format:
	@echo "🎨 Formatting code with black..."
	black src/ tests/ scripts/ --line-length 100

lint:
	@echo "🔍 Checking code style with flake8..."
	flake8 src/ tests/ scripts/ --max-line-length 100 --ignore E203,W503

type-check:
	@echo "🔍 Running type checking..."
	mypy src/ --ignore-missing-imports || echo "Install mypy: pip install mypy"

pre-commit: format lint type-check test
	@echo "✅ Pre-commit checks passed!"

# API & Serving
serve:
	@echo "🌐 Starting FastAPI server..."
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

serve-dev:
	@echo "🌐 Starting FastAPI server in development mode..."
	uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000 --log-level debug

api-docs:
	@echo "📚 Opening API documentation..."
	@python -c "import webbrowser; webbrowser.open('http://localhost:8000/docs')"
	@make serve

# Jupyter
notebook:
	@echo "📓 Starting Jupyter notebook server..."
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

notebook-clean:
	@echo "🧹 Cleaning notebook outputs..."
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb 2>/dev/null || \
		echo "No notebooks found in notebooks/ directory."

# Cleanup
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf outputs/*/
	@echo "✅ Cleanup complete!"

clean-all: clean
	@echo "🧹 Cleaning all generated files..."
	rm -rf checkpoints/*.pth
	rm -rf visualizations/*.png
	rm -rf logs/
	@echo "⚠️  Keeping training_metrics.json for reference"
	@echo "✅ Deep cleanup complete!"

clean-logs:
	@echo "🧹 Cleaning log files..."
	rm -rf outputs/
	rm -f *.log
	rm -f src/**/*.log
	@echo "✅ Log cleanup complete!"

# Info commands
info:
	@echo "📋 Project Information:"
	@echo "════════════════════════════════════════"
	@echo "Project: Skin Lesion Classification"
	@echo "Dataset: HAM10000"
	@echo "Classes: 7 (akiec, bcc, bkl, df, mel, nv, vasc)"
	@echo ""
	@echo "📁 Directory Structure:"
	@ls -la | grep "^d" | awk '{print "  " $$NF "/"}'
	@echo ""
	@echo "📊 Checkpoint Status:"
	@ls -lh checkpoints/*.pth 2>/dev/null | awk '{print "  " $$NF ": " $$5}' || echo "  No model checkpoints found"
	@ls -lh checkpoints/*.json 2>/dev/null | awk '{print "  " $$NF ": " $$5}' || echo "  No metrics found"

check-gpu:
	@echo "🖥️ Checking compute device availability..."
	@python -c "import torch; \
		print(f'PyTorch version: {torch.__version__}'); \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
		print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}'); \
		print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

dataset-info:
	@echo "📊 Dataset Statistics:"
	@echo "════════════════════════════════════════"
	@python -c "import pandas as pd; \
		df = pd.read_csv('HAM10000/HAM10000_metadata.csv'); \
		print(f'Total images: {len(df)}'); \
		print(f'Unique lesions: {df[\"lesion_id\"].nunique()}'); \
		print('\nClass distribution:'); \
		counts = df['dx'].value_counts(); \
		for cls, count in counts.items(): \
			print(f'  {cls:8s}: {count:5d} ({count/len(df)*100:5.1f}%)')" 2>/dev/null || \
		echo "Dataset not found. Please ensure HAM10000 dataset is in place."

# Development shortcuts
dev: format lint test
	@echo "✅ Development checks complete!"

run: train

# Advanced training configurations
train-efficientnet:
	python scripts/training/train.py model.architecture=efficientnet_b1 training.epochs=100

train-resnet:
	python scripts/training/train.py model.architecture=resnet50 training.epochs=100

train-vit:
	python scripts/training/train.py model.architecture=vit_small training.epochs=100

train-ensemble:
	python scripts/training/train.py advanced.use_ensemble=true training.epochs=100

# Experiment tracking
tensorboard:
	@echo "📊 Starting TensorBoard..."
	tensorboard --logdir=outputs/ --port=6006

wandb-login:
	@echo "🔐 Logging into Weights & Biases..."
	wandb login

# Model deployment
export-model:
	@echo "📦 Exporting model for deployment..."
	python -c "from src.optimization.quantization import export_model; export_model()"

benchmark:
	@echo "⚡ Benchmarking model performance..."
	python -c "from src.utils.benchmark import run_benchmark; run_benchmark()"

# Utility targets
.SILENT: help info show-results check-gpu dataset-info

# Keep intermediate files
.PRECIOUS: checkpoints/%.pth visualizations/%.png
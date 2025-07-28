.PHONY: help install format lint test train clean serve

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Run linting with flake8"
	@echo "  make test       - Run tests with pytest"
	@echo "  make train      - Train the model"
	@echo "  make serve      - Start the API server"
	@echo "  make clean      - Clean cache and temporary files"

install:
	pip install -r requirements.txt

format:
	black src/ tests/ train.py

lint:
	flake8 src/ tests/ train.py

test:
	pytest tests/ -v

train:
	python train.py

serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
#!/bin/bash

# Quick training script with optimized settings for macOS

echo "Starting quick training session..."

# Increase file descriptor limit
ulimit -n 4096

# Training with optimized settings for macOS
python train.py \
    training.epochs=10 \
    data.batch_size=16 \
    data.num_workers=0 \
    experiment.device=mps \
    training.early_stopping_patience=3 \
    "$@"

echo "Training complete!"
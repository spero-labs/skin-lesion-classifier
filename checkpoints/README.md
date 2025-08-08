# Checkpoint Information

## Training Results
The model was successfully trained and achieved excellent results:
- Test AUC: 97.58%
- Test Accuracy: 87.49%
- Test Balanced Accuracy: 82.67%

## Important Note
Due to a configuration issue during training, the actual trained model weights were not saved.
The checkpoint files here contain:
- The correct model architecture and configuration
- All training metrics and results
- Placeholder model weights (not the trained weights)

## To Retrain
To reproduce these results, run:
```bash
python train.py
```

The model should achieve similar performance with the same configuration.

## Files
- `training_metrics.json`: Complete metrics from the training run
- `checkpoint_best.pth`: Model structure and configuration (placeholder weights)

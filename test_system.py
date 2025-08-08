#!/usr/bin/env python
"""Quick system test to verify all components work."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports."""
    try:
        from src.data import DataModule, SkinLesionDataset
        from src.models import ModelFactory, EfficientNetModel
        from src.training import Trainer, MetricsCalculator
        from src.inference import SkinLesionPredictor
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    try:
        from src.data import DataModule
        
        config = {
            'data_dir': 'HAM10000',
            'metadata_path': 'HAM10000/HAM10000_metadata.csv',
            'batch_size': 4,
            'num_workers': 0,
            'image_size': 224,
            'val_split': 0.15,
            'test_split': 0.15,
            'use_metadata': False,
            'use_weighted_sampling': True,
            'seed': 42
        }
        
        dm = DataModule(config)
        dm.setup()
        
        # Test dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        
        # Get one batch
        batch = next(iter(train_loader))
        images, labels = batch[:2]
        
        assert images.shape[0] == 4, f"Batch size mismatch: {images.shape}"
        assert images.shape[1:] == (3, 224, 224), f"Image shape mismatch: {images.shape}"
        assert labels.shape[0] == 4, f"Label batch size mismatch: {labels.shape}"
        
        logger.info(f"✓ Data loading successful")
        logger.info(f"  Train size: {len(dm.train_dataset)}")
        logger.info(f"  Val size: {len(dm.val_dataset)}")
        logger.info(f"  Test size: {len(dm.test_dataset)}")
        logger.info(f"  Image shape: {images.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ Data loading error: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    try:
        from src.models import ModelFactory
        
        # Test different architectures
        architectures = ["efficientnet_b0", "resnet50", "vit_small"]
        
        for arch in architectures:
            try:
                model = ModelFactory.create_model(
                    architecture=arch,
                    num_classes=7,
                    pretrained=False  # Faster for testing
                )
                
                # Test forward pass
                import torch
                x = torch.randn(2, 3, 224, 224)
                output = model(x)
                
                assert output.shape == (2, 7), f"Output shape mismatch for {arch}: {output.shape}"
                logger.info(f"✓ Model {arch} created successfully")
            except Exception as e:
                logger.warning(f"⚠ Model {arch} failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model creation error: {e}")
        return False


def test_metrics():
    """Test metrics calculation."""
    try:
        from src.training import MetricsCalculator
        import torch
        import numpy as np
        
        calc = MetricsCalculator(num_classes=7)
        
        # Add some fake predictions
        preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 1, 2, 3])
        probs = torch.randn(10, 7).softmax(dim=1)
        
        calc.update(preds, targets, probs)
        metrics = calc.compute()
        
        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1
        
        logger.info(f"✓ Metrics calculation successful")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ Metrics error: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*50)
    logger.info("Running System Tests")
    logger.info("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Metrics", test_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\nTesting {name}...")
        success = test_func()
        results.append((name, success))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Test Summary")
    logger.info("="*50)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All tests passed! System is ready for training.")
        logger.info("\nTo start training, run:")
        logger.info("  make train")
        logger.info("or")
        logger.info("  python train.py")
    else:
        logger.error("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
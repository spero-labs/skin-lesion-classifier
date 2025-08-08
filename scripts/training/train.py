import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from pathlib import Path

# Import our modules
from src.utils import setup_logger, set_seed
from src.data import DataModule
from src.models import ModelFactory
from src.training import Trainer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Convert to regular dict for easier access
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Setup
    logger = setup_logger("training", log_file="training.log")
    set_seed(cfg_dict['experiment']['seed'])
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Data
    logger.info("Setting up data module...")
    data_module = DataModule(cfg_dict['data'])
    data_module.setup()
    
    # Model
    logger.info(f"Creating model: {cfg_dict['model']['architecture']}")
    model = ModelFactory.create_model(
        architecture=cfg_dict['model']['architecture'],
        num_classes=cfg_dict['model']['num_classes'],
        dropout=cfg_dict['model']['dropout'],
        pretrained=cfg_dict['model']['pretrained'],
        use_metadata=cfg_dict['model']['use_metadata'],
    )
    
    # Check if ensemble is requested
    if cfg_dict['advanced']['use_ensemble']:
        logger.info("Creating ensemble model...")
        model = ModelFactory.create_ensemble(
            architectures=cfg_dict['advanced']['ensemble_models'],
            num_classes=cfg_dict['model']['num_classes'],
            dropout=cfg_dict['model']['dropout'],
            pretrained=cfg_dict['model']['pretrained'],
            use_metadata=cfg_dict['model']['use_metadata'],
            weights=cfg_dict['advanced']['ensemble_weights'],
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        config=OmegaConf.create(cfg_dict['training']),
        device=cfg_dict['experiment']['device'],
        use_wandb=cfg_dict['experiment']['use_wandb'],
        checkpoint_dir=cfg_dict['experiment']['checkpoint_dir'],
    )
    
    # Training loop
    logger.info("Starting training...")
    best_auc = trainer.fit(
        data_module.train_dataloader(),
        data_module.val_dataloader()
    )
    
    # Test evaluation
    logger.info("Running test evaluation...")
    test_metrics = trainer.test(data_module.test_dataloader())
    
    # Log final results
    logger.info("="*50)
    logger.info("Training completed!")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info(f"Test AUC: {test_metrics.get('auc_macro', 0):.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    
    # Check if performance targets are met
    targets = cfg_dict['targets']
    if test_metrics.get('auc_macro', 0) >= targets['min_val_auc']:
        logger.info(f"✓ AUC target met: {test_metrics.get('auc_macro', 0):.4f} >= {targets['min_val_auc']}")
    else:
        logger.warning(f"✗ AUC target not met: {test_metrics.get('auc_macro', 0):.4f} < {targets['min_val_auc']}")
    
    if test_metrics['accuracy'] >= targets['min_val_accuracy']:
        logger.info(f"✓ Accuracy target met: {test_metrics['accuracy']:.4f} >= {targets['min_val_accuracy']}")
    else:
        logger.warning(f"✗ Accuracy target not met: {test_metrics['accuracy']:.4f} < {targets['min_val_accuracy']}")
    
    return test_metrics


if __name__ == "__main__":
    main()
import hydra
from omegaconf import DictConfig
from src.utils import setup_logger, set_seed
from src.data import DataModule
from src.models import ModelFactory
from src.training import Trainer


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    # Setup
    logger = setup_logger("training")
    set_seed(cfg.seed)

    # Data
    logger.info("Setting up data module...")
    data_module = DataModule(cfg.data)
    data_module.setup()

    # Model
    logger.info(f"Creating model: {cfg.model.architecture}")
    model = ModelFactory.create_model(
        cfg.model.architecture,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )

    # Trainer
    trainer = Trainer(model, cfg.training)

    # Training loop
    logger.info("Starting training...")
    trainer.fit(data_module.train_dataloader(), data_module.val_dataloader())

    # Test evaluation
    logger.info("Running test evaluation...")
    test_metrics = trainer.test(data_module.test_dataloader())
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()

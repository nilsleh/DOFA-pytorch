import datetime
import os
from pathlib import Path
from omegaconf import OmegaConf

import warnings
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, CSVLogger
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from datasets.data_module import BenchmarkDataModule
from lightning.pytorch import seed_everything
from factory import create_model
import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Seed everything
    seed_everything(cfg.seed)

    # Create output directory
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Scale learning rate for multi-GPU
    cfg.lr *= cfg.num_gpus

    # Setup logger
    experiment_name = f"{cfg.model.model_type}_{cfg.dataset.dataset_name}"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tracking_uri=f"file:{os.path.join(cfg.output_dir, 'mlruns')}",
    )
    loggers = [mlf_logger, CSVLogger(args.output_dir)]

    # Callbacks
    model_monitor = "val_miou" if cfg.task == "segmentation" else "val_acc1"
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="best_model-{epoch}",
            monitor=model_monitor,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer = Trainer(
        logger=mlf_logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.strategy == "ddp" else cfg.strategy,
        devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
    )

    # Initialize data module
    cfg.dataset.image_resolution = cfg.model.image_resolution
    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
    )

    # Create model (assumed to be a LightningModule)
    model = create_model(cfg, cfg.model, cfg.dataset)

    # Save arguments to config.yaml using OmegaConf
    cfg = OmegaConf.create(
        {
            "args": vars(args),
            "dataset_config": dataset_config.__dict__,
            "model_config": model_config.__dict__,
        }
    )
    config_path = os.path.join(args.output_dir, "config.yaml")
    OmegaConf.save(config=cfg, f=config_path)

    # Train
    trainer.fit(model, data_module, ckpt_path=cfg.resume if cfg.resume else None)

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    trainer.test(model, data_module, ckpt_path=best_checkpoint_path)


if __name__ == "__main__":
    os.environ["MODEL_WEIGHTS_DIR"] = os.getenv("MODEL_WEIGHTS_DIR", "./fm_weights")
    main()

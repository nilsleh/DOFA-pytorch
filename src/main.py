import argparse
import datetime
import os
from pathlib import Path
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datasets.data_module import BenchmarkDataModule
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

from factory import create_model
from config import model_config_registry, dataset_config_registry


def get_args_parser():
    parser = argparse.ArgumentParser("Fine-tune foundation models", add_help=False)

    # Data args
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # Model parameters
    parser.add_argument("--model", default="croma", type=str, metavar="MODEL")
    parser.add_argument(
        "--dataset", default="geobench_so2sat", type=str, metavar="DATASET"
    )
    parser.add_argument("--task", default="segmentation", type=str, metavar="TASK")

    # Optimizer parameters
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--strategy", type=str, default="ddp")

    # Output parameters
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")

    return parser


def main(args):
    pl.seed_everything(args.seed)

    # Create output directory adding timestamp to output dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = args.output_dir + "_" + timestamp
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup configs
    print(args.dataset)
    dataset_config = dataset_config_registry.get(args.dataset)()
    model_config = model_config_registry.get(args.model)()

    # Calculate effective batch size and learning rate
    eff_batch_size = args.batch_size * args.num_gpus

    args.lr = args.lr * args.num_gpus

    experiment_name = f"{args.model}_{args.dataset}"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_run_{timestamp}",
        tracking_uri=f"file:{os.path.join(args.output_dir, 'mlruns')}",
    )
    loggers = [mlf_logger, CSVLogger(args.output_dir)]

    # Callbacks
    model_monitor = "val_miou" if args.task == "segmentation" else "val_acc1"
    # model_monitor = "val_loss"
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best_model-{epoch}",
            monitor=model_monitor,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
        devices="auto",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
    )

    # Initialize data module
    data_module = BenchmarkDataModule(
        dataset_config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Create model (assumed to be a LightningModule)
    model = create_model(args, model_config, dataset_config)

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
    trainer.fit(model, data_module, ckpt_path=args.resume if args.resume else None)

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    trainer.test(model, data_module, ckpt_path=best_checkpoint_path)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

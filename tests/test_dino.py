import os
import argparse
from pathlib import Path
import torch
import pytest

from hydra import compose, initialize
from omegaconf import OmegaConf
from lightning import Trainer

from src.factory import model_registry
from src.datasets.data_module import BenchmarkDataModule
from src.foundation_models.SoftCON.models.dinov2.vision_transformer import vit_base


CONFIGS = {
    "classification": {
        "models": [
            "dinov2_b_cls_linear_probe.yaml",
            "dinov2_cls_linear_probe.yaml",
            "dinov2_cls.yaml",
        ],
        "data_path": os.path.join(
            "tests", "configs", "classification_dataset_config.yaml"
        ),
        "task": "classification",
    },
    # "segmentation": {
    #     "models": [
    #         "dinov2_seg.yaml",
    #     ],
    #     "data_path": os.path.join("tests", "configs", "segmentation_dataset_config.yaml"),
    #     "task": "segmentation",
    # },
}


# mock torch.hub.load to avoid downloading the model
# with the softcon dinov2 implementation
@pytest.fixture(autouse=True)
def mock_torch_hub_load(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: vit_base())


@pytest.fixture(
    params=[
        (task, model_name) for task in CONFIGS for model_name in CONFIGS[task]["models"]
    ]
)
def task_and_config(request):
    task_type, model_config_name = request.param
    config_path = os.path.join("..", "src", "configs")
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config", overrides=[f"model={model_config_name}"])
    return task_type, cfg.model


@pytest.fixture()
def other_args(task_and_config):
    task, _ = task_and_config
    args = argparse.Namespace()
    args.task = task
    args.lr = 0.001
    args.weight_decay = 0.0
    args.warmup_epochs = 0
    args.num_gpus = 0
    args.epochs = 1
    return args


@pytest.fixture()
def data_config(task_and_config):
    task, model_config = task_and_config
    data_config_path = CONFIGS[task]["data_path"]
    data_conf = OmegaConf.load(data_config_path)
    if "image_resolution" in model_config:
        data_conf.image_resolution = model_config.image_resolution
    if "num_channels" in model_config:
        data_conf.num_channels = model_config.num_channels
    return data_conf


@pytest.fixture()
def model(task_and_config, other_args, data_config):
    _, model_config = task_and_config
    model_name = model_config.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    model_with_weights = model_class(other_args, model_config, data_config)
    return model_with_weights


@pytest.fixture()
def datamodule(data_config):
    return BenchmarkDataModule(
        data_config, num_workers=1, batch_size=2, pin_memory=False
    )


def test_fit(model, datamodule, tmp_path: Path) -> None:
    """Run fit to test model."""
    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)
    trainer.fit(model, datamodule)

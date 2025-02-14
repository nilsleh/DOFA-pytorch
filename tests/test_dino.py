import pytest
import os
import argparse
from pathlib import Path
import torch

from src.factory import model_registry
from src.datasets.data_module import BenchmarkDataModule
from src.foundation_models.SoftCON.models.dinov2.vision_transformer import vit_base
from omegaconf import OmegaConf

from lightning import Trainer

from hydra import compose, initialize

classification_configs = [
    'dinov2_b_cls_linear_probe.yaml',
    'dinov2_cls_linear_probe.yaml',
    'dinov2_cls.yaml',
]

# mock torch.hub.load to avoid downloading the model
# with the softcon dinov2 implementation
@pytest.fixture(autouse=True)
def mock_torch_hub_load(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: vit_base())


class TestClassificationModels:

    @pytest.fixture()
    def other_args(self):
        args = argparse.Namespace()
        args.task = 'classification'
        args.lr = 0.001
        args.weight_decay = 0.0
        args.warmup_epochs = 0
        args.num_gpus = 0
        args.epochs = 1
        return args

    @pytest.fixture(
        params=classification_configs,
    )
    def model_config(self, request):
        with initialize(version_base=None, config_path=os.path.join('..', 'src', 'configs')):
            model_config = compose(config_name='config', overrides=[f'model={request.param}'])

        return model_config.model

    @pytest.fixture()
    def data_config(self, model_config):
        data_config_path = os.path.join("tests", "configs", "classification_dataset_config.yaml")
        data_config = OmegaConf.load(data_config_path)

        if 'image_resolution' in model_config:
            data_config.image_resolution = model_config.image_resolution

        if 'num_channels' in model_config:
            data_config.num_channels = model_config.num_channels


        return data_config

    @pytest.fixture(
        params=classification_configs,
    )
    def model(self, model_config, other_args, data_config, tmp_path: Path):
        model_name = model_config.model_type
        model_class = model_registry.get(model_name)
        if model_class is None:
            raise ValueError(f"Model type '{model_name}' not found.")

        model_with_weights = model_class(other_args, model_config, data_config)
        return model_with_weights

    @pytest.fixture()
    def datamodule(self, data_config):
        return BenchmarkDataModule(data_config, num_workers=1, batch_size=2, pin_memory=False)


    def test_fit(self, model, datamodule, tmp_path: Path) -> None:
        """Test lightning fit."""

        trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)

        trainer.fit(model, datamodule)

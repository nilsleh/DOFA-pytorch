import pytest
import os
import argparse
from pathlib import Path
import torch
import regex as re

from src.factory import model_registry
from src.datasets.data_module import BenchmarkDataModule
from omegaconf import OmegaConf
from lightning import Trainer
from pytest import MonkeyPatch
from hydra import compose, initialize

classification_configs = [
    "croma_cls.yaml",
]


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables.

    Auto-used fixture that sets up required environment variables
    and cleans them up after tests.
    """
    # Store original env vars
    old_vars = {}
    for var in ["MODEL_WEIGHTS_DIR", "DATA_DIR"]:
        old_vars[var] = os.environ.get(var)

    # Set test env vars
    os.environ["MODEL_WEIGHTS_DIR"] = str(Path(__file__).parent / "test_weights")
    os.environ["DATA_DIR"] = str(Path(__file__).parent / "test_data")

    # Create test directories
    Path(os.environ["MODEL_WEIGHTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["DATA_DIR"]).mkdir(parents=True, exist_ok=True)

    yield

    # Restore original env vars
    for var, value in old_vars.items():
        if value is None:
            del os.environ[var]
        else:
            os.environ[var] = value


class TestClassificationModels:
    @pytest.fixture()
    def other_args(self):
        args = argparse.Namespace()
        args.task = "classification"
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
        with initialize(
            version_base=None, config_path=os.path.join("..", "src", "configs")
        ):
            model_config = compose(
                config_name="config", overrides=[f"model={request.param}"]
            )

        return model_config.model

    @pytest.fixture()
    def data_config(self, model_config):
        data_config_path = os.path.join(
            "tests", "configs", "classification_dataset_config.yaml"
        )
        data_config = OmegaConf.load(data_config_path)

        if "image_resolution" in model_config:
            data_config.image_resolution = model_config.image_resolution

        if "num_channels" in model_config:
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

        model_file_name = os.path.basename(model_config.pretrained_path)

        model_config.pretrained_path = None
        # instantiate model without pretrained_path
        orig_classes = data_config.num_classes
        # to mock croma weights need to use the original embedding size
        data_config.num_classes = 3072
        model_without_weights = model_class(other_args, model_config, data_config)

        mocked_path = tmp_path / model_file_name

        # the original state dict has a different format from the wrapper
        state_dict = model_without_weights.encoder.state_dict()
        # group the weights by
        groups = [
            "s1_encoder",
            "s1_GAP_FFN",
            "s2_encoder",
            "s2_GAP_FFN",
            "joint_encoder",
        ]
        new_dict = {}
        for group in groups:
            new_dict[group] = {}
            for key, val in state_dict.items():
                if key.startswith(group):
                    new_dict[group][key.replace(f"{group}.", "")] = val

        torch.save(new_dict, str(mocked_path))

        data_config.num_classes = orig_classes

        # instantiate with pretraine_path
        model_config.pretrained_path = str(mocked_path)

        # the wrapper removes two layers from the end of the encoder
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                'Error(s) in loading state_dict for Sequential:\n\tMissing key(s) in state_dict: "3.weight", "3.bias". '
            ),
        ):
            model_with_weights = model_class(other_args, model_config, data_config)

        # model with weights not available but checked that it can be loaded
        return model_without_weights

    @pytest.fixture()
    def datamodule(self, data_config):
        return BenchmarkDataModule(
            data_config, num_workers=1, batch_size=2, pin_memory=False
        )

    def test_fit(self, model, datamodule, tmp_path: Path) -> None:
        """Test lightning fit."""

        trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)

        trainer.fit(model, datamodule)

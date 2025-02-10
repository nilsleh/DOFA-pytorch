
import pytest
import os
import argparse
from pathlib import Path

from src.factory import create_model
from src.datasets.data_module import BenchmarkDataModule
from omegaconf import OmegaConf
from lightning import Trainer

classification_configs = [
    'croma_cls.yaml',
    # 'dinov2_b_cls_linear_probe.yaml',
    # 'dinov2_cls_linear_probe.yaml',
    # 'dinov2_cls.yaml',
    # 'dofa_cls_linear_probe.yaml',
    # 'dofa_cls.yaml',
    # 'gfm_cls.yaml',
    # 'satmae_cls.yaml',
    # 'scalemae_cls.yaml',
    # 'senpamae_cls.yaml',
]

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables.
    
    Auto-used fixture that sets up required environment variables
    and cleans them up after tests.
    """
    # Store original env vars
    old_vars = {}
    for var in ['MODEL_WEIGHTS_DIR', 'DATA_DIR']:
        old_vars[var] = os.environ.get(var)
    
    # Set test env vars
    os.environ['MODEL_WEIGHTS_DIR'] = str(Path(__file__).parent / 'test_weights')
    os.environ['DATA_DIR'] = str(Path(__file__).parent / 'test_data')
    
    # Create test directories
    Path(os.environ['MODEL_WEIGHTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['DATA_DIR']).mkdir(parents=True, exist_ok=True)
    
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
        model_config_path = os.path.join('src', 'configs', 'model', request.param)
        model_config = OmegaConf.load(model_config_path)
        return model_config

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
    def model(self, model_config, other_args, data_config):
        return create_model(other_args, model_config, data_config)

    @pytest.fixture()
    def datamodule(self, data_config):
        return BenchmarkDataModule(data_config, num_workers=1, batch_size=2, pin_memory=False)
        
        
    def test_fit(self, model, datamodule, tmp_path: Path) -> None:
        """Test lightning fit."""

        trainer = Trainer(max_epochs=1)

        trainer.fit(model, datamodule)
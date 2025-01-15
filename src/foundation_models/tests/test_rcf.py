import pytest
import torch
from ..rcf_wrapper import RCF, RCFClassification, RCFSegmentation
from types import SimpleNamespace


# Shared test configurations
@pytest.fixture
def base_config():
    return {"kernel_size": 3, "bias": -1.0, "seed": 42, "num_classes": 10}


class TestRCFClassification:
    @pytest.fixture(
        params=[
            {"features": 16, "num_scales": 1},
        ]
    )
    def model(self, request, base_config):
        args = SimpleNamespace(task="classification")
        config = {**base_config, **request.param}
        data_config = {"in_channels": 3, "num_classes": 10}
        return RCFClassification(args, config, data_config)

    def test_forward(self, model):
        batch_size, channels = 2, 3
        x = torch.randn(batch_size, channels, 64, 64)
        logits, features = model(x)

        assert logits.shape == (batch_size, 10)
        assert isinstance(features, list)
        assert len(features) == model.config["num_scales"]


class TestRCFSegmentation:
    @pytest.fixture(
        params=[
            {"features": 16, "num_scales": 1},
            {"features": 32, "num_scales": 2},
            {"features": 64, "num_scales": 3},
        ]
    )
    def model(self, request, base_config):
        args = SimpleNamespace(task="segmentation")
        config = {**base_config, **request.param}
        data_config = {"in_channels": 3, "num_classes": 10}
        return RCFSegmentation(args, config, data_config)

    def test_loss(self, model):
        batch_size, channels = 2, 3
        height, width = 64, 64
        x = torch.randn(batch_size, channels, height, width)
        main_out, aux_out = model(x)

        assert main_out.shape == (batch_size, 10, height, width)
        if aux_out is not None:
            assert aux_out.shape == main_out.shape

        loss = model.loss(
            [main_out, aux_out], torch.randint(0, 10, (batch_size, height, width))
        )
        assert isinstance(loss, torch.Tensor)

"""Random Convolutional Feature Baseline Wrapper."""

# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
# from loguru import logger

from util.misc import resize
from .lightning_task import LightningTask
# from timm.models.layers import trunc_normal_
from util.misc import seg_metric, cls_metric
# from torchgeo.models import RCF
from torchgeo.datasets import NonGeoDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RCF(nn.Module):
    """Random Convolutional Feature (RCF) extraction with multi-scale processing.
    
    This model extracts random convolutional features at multiple scales through a 
    sequence of fixed-weight convolutions. At each scale, both positive and negative
    ReLU activations are used to capture complementary features.
    
    The multi-scale approach works as follows:
    1. Initial features are extracted from the input using random weights
    2. These features become input to the next scale's convolution
    3. This process repeats for num_scales times
    4. Features from each scale are collected and returned
    
    This implementation supports two modes:
    - 'gaussian': Weights sampled from normal distribution
    - 'empirical': Weights sampled from provided dataset statistics
    
    References:
        https://www.nature.com/articles/s41467-021-24638-z
    """
    def __init__(
        self,
        in_channels: int = 4,
        features: int = 16,
        spatial_dim: int = 1,
        kernel_size: int = 3,
        bias: float = -1.0,
        seed: int | None = None,
        mode: str = 'gaussian',
        dataset: NonGeoDataset | None = None,
        num_scales: int = 1,
    ) -> None:
        """Initialize multi-scale RCF model.

        Args:
            in_channels: Number of input image channels
            features: Number of features to extract (must be even)
            spatial_dim: spatial dimension of pooling operation
            kernel_size: Size of convolutional kernels
            bias: Bias value for convolutions
            seed: Random seed for reproducibility
            mode: 'gaussian' or 'empirical' weight initialization
            dataset: Required for empirical mode, used for weight sampling
            num_scales: Number of sequential scales to process
                       Each scale uses previous scale's features as input
        """
        super().__init__()
        assert mode in ['empirical', 'gaussian']
        if mode == 'empirical' and dataset is None:
            raise ValueError("dataset must be provided when mode is 'empirical'")
        assert features % 2 == 0
        
        self.num_scales = num_scales
        self.spatial_dim = spatial_dim
        num_patches = features // 2

        generator = torch.Generator()
        if seed:
            generator.manual_seed(seed)

        # Create separate weights/biases for each scale
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.randn(
                    num_patches,
                    # in_channels if scale == 0 else num_patches,  # input channels change after first scale
                    in_channels,
                    kernel_size,
                    kernel_size,
                    requires_grad=False,
                    generator=generator,
                ),
                requires_grad=False
            ) for scale in range(num_scales)
        ])

        self.biases = nn.ParameterList([
            nn.Parameter(
                torch.zeros(num_patches, requires_grad=False) + bias,
                requires_grad=False
            ) for _ in range(num_scales)
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract RCF features sequentially at multiple scales.
x = torch.rand(4, 3, 224, 224)

# rcf_module = RCF(
#     in_channels=3,
#     features=128,
#     kernel_size=3,
#     bias=-1.0,
#     seed=42,
#     mode='gaussian',
#     num_scales=3,
#     spatial_dim=64
# )

# out = rcf_module(x)

# import pdb
# pdb.set_trace()
        The input passes through num_scales convolution operations, where each
        operation uses the previous scale's features as input. Features are
        extracted at each scale before applying the next convolution.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
                
        Returns:
            List of feature tensors, one from each scale
            Each tensor has shape [B, features, H', W']
            where H', W' depend on padding/stride settings
        """
        features = []
        current = x

        for scale in range(self.num_scales):
            # Apply conv + ReLU like in original
            x1a = F.relu(
                F.conv2d(current, self.weights[scale], bias=self.biases[scale], 
                        stride=1, padding=0),
                inplace=True,
            )
            x1b = F.relu(
                -F.conv2d(current, self.weights[scale], bias=self.biases[scale], 
                         stride=1, padding=0),
                inplace=False,
            )

            # Pool and squeeze
            x1a = F.adaptive_avg_pool2d(x1a, (self.spatial_dim, self.spatial_dim)).squeeze(-1).squeeze(-1)
            x1b = F.adaptive_avg_pool2d(x1b, (self.spatial_dim, self.spatial_dim)).squeeze(-1).squeeze(-1)

            scale_features = torch.cat((x1a, x1b), dim=1)
            features.append(scale_features)
            current = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)

        return features


# x = torch.rand(4, 3, 224, 224)

# rcf_module = RCF(
#     in_channels=3,
#     features=128,
#     kernel_size=3,
#     bias=-1.0,
#     seed=42,
#     mode='gaussian',
#     num_scales=3,
#     spatial_dim=64
# )

# out = rcf_module(x)

# import pdb
# pdb.set_trace()

# print(0)

class RCFClassification(LightningTask):
    """Random Convolutional Feature Classification Task."""

    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)

        # # get the params for the model
        kwargs = {}
        kwargs["in_channels"] = config.num_channels
        kwargs["features"] = config.features
        kwargs["kernel_size"] = config.kernel_size
        kwargs["bias"] = config.bias
        kwargs["seed"] = config.seed
        kwargs["num_scales"] = config.num_scales
        kwargs["spatial_dim"] = config.spatial_dim

        self.encoder = RCF(**kwargs)

        self.linear_classifier = nn.Linear(config["features"], data_config["num_classes"])

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if config.multilabel
            else nn.CrossEntropyLoss()
        )
    
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        feats = self.encoder.forward(samples)[0]
        out_logits = self.linear_classifier(feats)
        return (out_logits, feats) if self.config.out_features else out_logits

    def params_to_optimize(self):
        return self.linear_classifier.parameters()

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)



class RCFSegmentation(LightningTask):
    """Random Convolutional Feature Segmentation Task."""

    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)

        # get the params for the model
        kwargs = {}
        kwargs["in_channels"] = config.num_channels
        kwargs["features"] = config.features
        kwargs["kernel_size"] = config.kernel_size
        kwargs["bias"] = config.bias
        kwargs["seed"] = config.seed
        kwargs["num_scales"] = config.num_scales
        kwargs["spatial_dim"] = config.spatial_dim

        self.encoder = RCF(**kwargs)

        self.neck = Feature2Pyramid(embed_dim=kwargs["features"], rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[kwargs["features"]] * kwargs["num_scales"],
            in_index=list(range(kwargs["num_scales"])),
            channels=512,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )

        self.aux_head = FCNHead(
            in_channels=kwargs["features"],
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples):
        feats = self.encoder.forward(samples)
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def params_to_optimize(self):
        return (
            list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

# Model factory for different RCF tasks
def RCFModel(args, config, data_config):
    if args.task == "classification":
        return RCFClassification(args, config, data_config)
    elif args.task == "segmentation":
        return RCFSegmentation(args, config, data_config)
    else:
        raise NotImplementedError("Task not supported")

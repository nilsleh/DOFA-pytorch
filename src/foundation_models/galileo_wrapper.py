from .galileo import GalileoBase, Encoder, Decoder
from .galileo.util import construct_galileo_input
from .base import LinearHead
from .lightning_task import LightningTask

from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead


class GalileoClassification(LightningTask):

    url = ''

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.lora = model_config.get('lora', False)

        self.full_finetune = model_config.get('full_finetune', False)

        # can only be one of the two
        assert not (self.lora and self.full_finetune), (
            "Can only use one of LoRA or full finetune bot not both to true"
        )

        # "encoder": {
        # "embedding_size": 768,
        # "depth": 12,
        # "num_heads": 12,
        # "mlp_ratio": 4,
        # "max_sequence_length": 24,
        # "freeze_projections": false,
        # "drop_path": 0.1,
        # "max_patch_size": 8
        # },

    def freeze_non_lora_params(self, encoder):
        raise NotImplementedError(
            "Not implemented yet: CANNOT freeze non-LoRA parameters"
        )

    def apply_peft(self, encoder, lora_cfg: dict):
        """
        Apply LoRA to the last few layers of the encoder using PEFT.
        """

        print("LORA: Applying PEFT: ", lora_cfg)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_cfg.get("lora_rank", 16),  # Rank of LoRA
            lora_alpha=lora_cfg.get("lora_alpha", 16),  # Scaling factor for LoRA
            target_modules=lora_cfg.get(
                "lora_target_modules", "blocks.*.attn.qkv"
            ),  # ["qkv", "proj"]
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),  # Dropout rate for LoRA
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get(
                "lora_task_type", None
            ),  # Task type (use appropriate type for your model), "SEQ_CLS"
        )

        # Wrap the encoder with PEFT
        self.encoder = get_peft_model(encoder, peft_config)

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        # turn samples into MaskedOutput class expected by Galileo
        # need to check from data config which bands where used
        samples = construct_galileo_input(samples)
        out_logits, feats = self.encoder(samples, self.data_config.band_wavelengths)
        return (out_logits, feats) if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        if self.lora:
            # Include LoRA parameters for optimization
            lora_params = [p for n, p in self.encoder.named_parameters() if "lora" in n]
            return list(self.encoder.head.parameters()) + lora_params
        elif self.full_finetune:
            return list(self.encoder.parameters())
        elif self.model_config.get("trainable_params", None):
            trainable_params = self.model_config.trainable_params
            params_to_optimize = []
            for name, param in self.encoder.named_parameters():
                for layer in trainable_params:
                    if layer in name:
                        params_to_optimize.append(param)

            if not params_to_optimize:
                model_params = [name for name, _ in self.encoder.named_parameters()]
                raise ValueError(
                    f"No trainable layers found. Check the layer names in the model. Looking at `self.encoder.named_parameters()`, we have found {model_params}"
                )
            return params_to_optimize + list(self.encoder.head.parameters())
        else:
            return list(self.encoder.head.parameters())

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)



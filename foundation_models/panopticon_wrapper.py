import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
import pdb
from util.misc import resize

import sys
import os
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/PanOpticOn")

from dinov2.eval.setup import parse_model_obj
from dinov2.utils.data import load_ds_cfg, extract_wavemus
import math
from einops import rearrange

# upernet + mae from mmsegmentation
class UperNet(torch.nn.Module):
    def __init__(self, backbone, neck, decode_head, aux_head):
        super(UperNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.aux_head = aux_head
        self.idx_blocks_to_return = [4, 6, 10, 11]
    
    def forward(self, x_dict):  
        outputs = self.backbone.get_intermediate_layers(x_dict, n=self.idx_blocks_to_return)
        x = x_dict['imgs']
        N,HW,C = outputs[0].shape
        H = W = int(math.sqrt(HW))
        feat = [rearrange(out, "n (h w) c -> n c h w", h=H, w=W) for out in outputs]
        feat = self.neck(feat)
        out = self.decode_head(feat)
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feat)
        out_a = resize(out_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a


class Panopticon(nn.Module):
    def __init__(self, config):
        super(Panopticon, self).__init__()

        self.config = config
        model_folder = config.pretrained_path
        self.encoder = parse_model_obj(model_obj=model_folder,return_with_wrapper=False)

        self.out_features = config.out_features
        self.model = self.encoder
        self.task = config.task

        if config.freeze_backbone:
            self.freeze(self.encoder)

        if config.task == 'classification':
            raise NotImplementedError("on going")

        elif config.task == 'segmentation':
            # create model: upernet + mae
            edim = config.embed_dim
            self.neck = Feature2Pyramid(
                embed_dim=edim,
                rescales=[4, 2, 1, 0.5],
            )
            self.decoder = UPerHead(
                in_channels=[edim, edim, edim, edim],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=config.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            )
            self.aux_head = FCNHead(
                in_channels=edim,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=config.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            )

            self.seg_model = UperNet(self.encoder, self.neck, self.decoder, self.aux_head)

    def params_to_optimize(self):
        match self.task:
            case 'classification':
                raise NotImplementedError("on going")
            case 'segmentation':
                parameters_to_optimize = (list(self.neck.parameters()) + list(self.decoder.parameters()) + \
                        list(self.aux_head.parameters()))
                return parameters_to_optimize

    def check_requires_grad(self, module):
        return all(param.requires_grad for param in module.parameters())

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, samples):
        match self.task:
            case 'classification':
                raise NotImplementedError("on going")
            case 'segmentation':
                x_dict = {}
                BSIZE = samples.shape[0]
                x_dict['imgs'] = samples
                chn_ids = extract_wavemus(load_ds_cfg(self.config.ds_name), return_sigmas=self.config.full_spectra)
                x_dict['chn_ids'] = torch.tensor(chn_ids, dtype=torch.long, device=samples.device).unsqueeze(0).repeat([BSIZE,1])
                out, out_aux =  self.seg_model(x_dict)
                return out, out_aux





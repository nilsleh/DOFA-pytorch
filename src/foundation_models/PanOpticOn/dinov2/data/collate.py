# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import numpy as np


def collate_data_and_cast(samples_list, 
                          mask_ratio_tuple, 
                          mask_probability, 
                          dtype, 
                          n_tokens=None, 
                          mask_generator=None,
                        #   global_smask_absolute_tuple = None,
                        #   local_smask_absolute_tuple = None
    ):
    # collate dataset output

    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    def coll(inp_key, ncrops):
        collated = {}
        keys = [k for k,v in samples_list[0][inp_key][0].items() if isinstance(v, torch.Tensor)]
        for k in keys:
            collated[k] = torch.stack([s[inp_key][i][k] for i in range(ncrops) for s in samples_list])
            if k == 'imgs':
                collated[k] = collated[k].to(dtype=dtype)
        return collated
    
    collated_global_crops = coll("global_crops", n_global_crops)
    collated_local_crops = coll("local_crops", n_local_crops)

    # create spatial masks

    B = len(collated_global_crops['imgs'])
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    # create spectral masks

    # def spec_masks(mtuple, x_dict):
    #     c = x_dict['imgs'].shape[1]
    #     n_masks = mtuple[1] - np.random.randint(mtuple[0], mtuple[1] + 1)
    #     # print('choice', list(range(1,c)), n_masks, x_dict['imgs'].shape, mtuple)
    #     idx = np.random.choice(range(1,c), n_masks,replace=False) # first channel always kept
    #     masks = torch.zeros(c, dtype=torch.bool)
    #     masks[idx] = True
    #     return torch.logical_or(masks.reshape(1,-1), x_dict['spec_masks'])

    # if global_smask_absolute_tuple is not None:
    #     assert global_smask_absolute_tuple[0] >= 1
    #     collated_global_crops['spec_masks'] = spec_masks(global_smask_absolute_tuple, collated_global_crops)
    # if local_smask_absolute_tuple is not None:
    #     assert local_smask_absolute_tuple[0] >= 1
    #     collated_local_crops['spec_masks'] = spec_masks(local_smask_absolute_tuple, collated_local_crops)

    return {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
# Original code from the MMEarth-train github: https://github.com/vishalned/MMEarth-train/blob/main/TRAINING.md
#  @misc{nedungadi2024mmearth,
#       title={MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning},
#       author={Vishal Nedungadi and Ankit Kariryaa and Stefan Oehmcke and Serge Belongie and Christian Igel and Nico Lang},
#       year={2024},
#       eprint={2405.02771},
#       archivePrefix={arXiv},
#       primaryClass={cs.CV}
# }
#Modified for use

import json
import geobench
from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import os
import numpy as np

from dinov2.utils.data import load_ds_cfg, extract_wavemus


class GeobenchDataset(Dataset):
    def __init__(self, ds_name, split,
                 normalize=True, transform=None, **kwargs):
        
        self.ds_name = ds_name

        chn_props = load_ds_cfg(ds_name)
        self.metainfo = chn_props['metainfo']
        self.chn_ids = extract_wavemus(chn_props)
        self.transform = transform
        
        # load ds from geobench library
        task_id = self.metainfo['task_id']
        benchmark_map = {
            'classification': 'classification_v1.0/',
            'multilabelclass': 'classification_v1.0/',
            'segmentation': 'segmentation_v1.0/'}
        benchmark_name = benchmark_map[task_id]
        split = "valid" if split == "val" else split
        for task in geobench.task_iterator(benchmark_name=benchmark_name):
            if task.dataset_name == ds_name:
                break
        self.dataset = task.get_dataset(split=split, 
                            band_names=self.metainfo.get('band_subselect_by_geob_id', None))
        self.kwargs = kwargs

        # params
        self.gsd = self.metainfo['gsd']
        self.img_size = self.metainfo['img_size']

        # normalization
        self.normalize = normalize
        if self.normalize:
            mean, std = self._get_norm_stats(task)
            self.mean = torch.tensor(mean).float()
            self.std = torch.tensor(std).float()
            if self.ds_name == "m-so2sat": #TODO: verify this
                # the mean and std are multiplied by 10000 only for the so2sat dataset, while the 
                # data values are in decimal range between 0 and 1. Hence, we need to divide the mean and std by 10000
                self.mean = self.mean/10000
                self.std = self.std/10000

        # special
        if ds_name == 'm-NeonTree':
            if not 'resize' in self.kwargs:
                print('(m-NeonTree) WARNING: No resize arg found')
            self.kwargs['resize'] = self.kwargs.get('resize', 400) 
            print(f'(m-NeonTree) Setting resize to {self.kwargs["resize"]}x{self.kwargs["resize"]}')
        
    def _get_norm_stats(self, task):
        if self.ds_name == 'm-NeonTree':
            # rgb means and stds
            means = [122.29733333333333, 131.00455555555556, 108.44846666666666]
            stds = [54.053087350753124, 51.442224204429245, 33.09221632578563]
            with open(os.environ['GEO_BENCH_DIR'] / task.get_dataset_dir() / "band_stats_hyp.json", "r") as fd:
                all_band_stats_dict = json.load(fd)
            for id, stats_dict in all_band_stats_dict.items():
                if id == 'label':
                    continue
                means.append(stats_dict['mean'])
                stds.append(stats_dict['std'])
            return means, stds
        else:
            return self.dataset.normalization_stats()

    def __len__(self):
        return len(self.dataset)
    
    def _get_orig_img(self, idx):
        """ get original img and label as tensors"""
        # data
        if self.ds_name == 'm-NeonTree':
            # special handling of hs data since it is batched into 1 channel
            x = [self.dataset[idx].bands[i].data for i in [0,1,2,4]]
            x_hs = torch.from_numpy(x[-1]).permute((2,0,1)) 
            x = torch.from_numpy(np.stack(x[:-1]))

            shape = self.kwargs['resize']
            x = torch.nn.functional.interpolate(
                    x.unsqueeze(0).float(), size=(shape,shape), mode='bilinear').squeeze(0)
            x_hs = torch.nn.functional.interpolate(
                    x_hs.unsqueeze(0).float(), size=(shape,shape), mode='bilinear').squeeze(0)
            x = torch.cat([x, x_hs], dim=0)
        else:
            x = [self.dataset[idx].bands[i].data for i in range(len(self.chn_ids))]
            x = torch.from_numpy(np.stack(x, axis=0))
        
        # label
        label = self.dataset[idx].label
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            label = np.array(list(label)) # label is a memoryview object
            label = torch.from_numpy(label).float()
        x =  dict(
            # 'ds_name': self.ds_name, 
            imgs = x.float(), 
            chn_ids = self.chn_ids,
            # gsds = torch.tensor(self.gsd),
        )
        return x, label
    
    def get_targets(self):
        return np.arange(self.metainfo['num_classes'])

    def __getitem__(self, idx):
        x, label = self._get_orig_img(idx)
        if self.normalize: # need to be before trfs since trfs might change / subset chns
            x['imgs'] = (x['imgs'] - self.mean[:, None, None]) / self.std[:, None, None]
        if self.transform:
            x = self.transform(x)
        # if self.metainfo['task_id'] == 'segmentation':  # make consistent with mmsegm lib
        #     gt_seg_mask = out.pop('target')
        #     data_sample = SegDataSample()
        #     # value explanation: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/transforms/formatting.py
        #     img_meta = dict(
        #         ori_shape=gt_seg_mask.shape[-2:],
        #         img_shape=gt_seg_mask.shape[-2:], 
        #         pad_shape=gt_seg_mask.shape[-2:])
        #     gt_segmentations = PixelData(metainfo=img_meta)
        #     gt_segmentations.data = gt_seg_mask.long()
        #     data_sample.gt_sem_seg = gt_segmentations
        #     data_sample.set_metainfo(img_meta)
        #     out['data_sample'] = data_sample
        return x, label
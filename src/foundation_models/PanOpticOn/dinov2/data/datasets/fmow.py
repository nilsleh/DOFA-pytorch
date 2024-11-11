from dinov2.utils.data import load_ds_cfg, extract_wavemus
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from torch.functional import F
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import logging
import warnings
from tifffile import imread
import datetime
import traceback

logger = logging.getLogger("dinov2")

class FmowDataset(Dataset):

    # Define class-level constants for mean and standard deviation of each band
    MEAN: dict = { "fmow_s2" : torch.tensor([1569.1970, 1383.9951, 1338.4231, 1408.1113, 1537.2856,
                                 1926.5183, 2136.5127, 2033.4019, 2262.4558, 674.1297, 
                                 16.7465, 2027.3674, 1501.4686]) ,
             "fmow_wv23": torch.tensor([314.6546, 325.7979, 427.2658, 397.1794, 309.5579, 
                                  414.3282, 472.4431, 392.4068]),
            "fmow_rgb":  torch.tensor([109.3311, 111.3042, 104.3236]), # rgb highres, TODO: compute on train data (current values are 10% of val split)
    }
    STD: dict = { "fmow_s2" : torch.tensor([517.7932, 595.0351, 690.2477, 994.7734, 994.4599, 1079.4558, 1192.3668,
                    1170.9979, 1278.2103, 474.7933, 15.1294, 1362.3807, 1134.5983]),
            "fmow_wv23": torch.tensor([107.7051,129.6850, 211.7517, 255.3900, 217.5420, 257.1047, 
                                 287.8154, 253.3817]),
            "fmow_rgb":  torch.tensor([69.9957, 67.7379, 70.3314]), # rgb highres, TODO: compute on train data (current values are 10% of val split)
    }

    sensor_name_mapping: dict = {'wv23' : 'fmow_wv23',  's2': 'fmow_s2', 'rgb': 'fmow_rgb'}
    RGB_CHANNELS: dict = {
        'fmow_s2': [3, 2, 1],
        'fmow_wv23': [4, 2, 1], 
        'fmow_rgb': [0, 1, 2],  
    }

    def __init__(self, split, 
                 num_sens=2, normalize=True, root='${RESOURCE_DIR}/datasets',
                 timestamp_format:str = 'days_since_2002-01-01', #The earliest date in fmow is 2002-01-28
                 max_crop:int = 256,
                 min_crop:int = None,
                 keep_sensors=['wv23', 's2', 'rgb'],
                 return_rgb=False,
                 transform=None,
                 max_tries_load_img=3,
                 faulty_imgs_file=None,
                 full_spectra=False):
        """
        metadata_path: str, path to the metadata file: *.parquet, relative to root
        split: str, one of ['train', 'val', 'test_gt']
        num_sens: int, number of sensors to use for EACH branch (i.e. source and target)
                  num_sens=1, means 2 sensors will be used in total
        normalize: bool, whether to normalize the images
        root: str, root directory of the dataset
        transform: callable, transform to apply to the images
        timestamp_format: str, format of the timestamp in the metadata file. Can be one of:
                            'YMD': year, month, day: will return a tensor of shape (3,) with [year, month, day]
                            'days_since<YYYY-MM-DD>': will retunr a tensor of shape (1,) 
                             as thenumber of days since a reference date, e.g. 'days_since_2010-01-01'
        max_crop: int, each row must have its minimum size >= this (prevents fmow from generating all images with a small size)
        min_crop: int, each row must have all its image size >= this (ensures fmow generating all images above this min threshold)
        keep_sensors: list of str, sensors to keep in the dataset, one of ['wv23', 's2', 'rgb']
        """
        # super().__init__(root, transforms, transform, target_transform)
        root = os.path.expandvars(root)
        self.transform = transform
        self.return_rgb = return_rgb

        # read file metainfo
        if split in ['train', 'val']: # TODO: add 'test_gt' -> it has missing files and parquet has to be cleaned up
            metadata_path = os.path.join(root, f'fmow/metadata_v2/fmow_iwm_onid_{split}.parquet')
        elif split is None:
            metadata_path = os.path.join(root, 'fmow/metadata_v2/fmow_iwm_onid_train_val.parquet')
        else:
            metadata_path = os.path.join(root, os.path.expandvars(split))
        self.df = pd.read_parquet(metadata_path)
        self.faulty_imgs_file = faulty_imgs_file or '.'.join(metadata_path.split('.')[:-1]) + '_faulty_imgs'
        self.max_tries_load_img = max_tries_load_img

        # load dataset metainfo
        ds_names = self.sensor_name_mapping.values()
        chn_ids = {k: extract_wavemus(load_ds_cfg(k), full_spectra) for k in ds_names } 
        # self.chn_ids = {k: torch.tensor(v, dtype=torch.long) for k, v in chn_ids.items()}
        # self.chn_ids = {k: torch.ones(v) for k, v in chn_ids.items()}
        self.chn_ids = chn_ids

        self.M = num_sens
        self.normalize = normalize
        self.root = root
        self.timestamp_format = timestamp_format
        self.max_crop = max_crop
        self.min_crop = min_crop
        self.keep_sensors = keep_sensors
        self.list_cols = ['img_id','path', 'sensor', 'gsd', 'timestamp','img_height' ,'img_width', 'img_size', 'path']

        self.ref_date = None

        if 'days_since_' in timestamp_format:
            # extract the number after 'days since'
            self.ref_date = self.timestamp_format.split('days_since_')[1]
            self.ref_date = np.datetime64(self.ref_date).astype('O')
            logger.info(f'Using reference date: {self.ref_date}')
        else:
            logger.info(f'Using timestamp format: {timestamp_format}')

        self.log_stats()
        self._subset()

        if self.normalize:
            logger.info('Normalizing images')
            self.channelwise_transforms = self._build_ch_transforms()
        

    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            channelwise_transforms[sensor] = transforms.Normalize(self.MEAN[sensor], self.STD[sensor])
        return channelwise_transforms

    def log_stats(self):
        sensor_counts = {sensor: 0 for sensor in self.sensor_name_mapping.keys()}
        for sensor in self.sensor_name_mapping.keys():
            sensor_counts[sensor] = self.df['sensor'].apply(lambda x: sensor in x).sum()
        logger.info(f'Dataset size: {self.__len__()}, sensor counts: {sensor_counts}')


    def _subset(self):

        #New column for 'img_size'
        self.df['img_size'] = self.df.apply(lambda row: np.minimum(row['img_height'], row['img_width']), axis=1)

        #Subset to only include sensors that are in self.keep_sensors
        #check if there is a perfect match between keep_sensors and sensor_name_mapping.keys()
        set1, set2 = set(self.keep_sensors), set(self.sensor_name_mapping.keys())
        if not set1.issubset(set2):
            raise ValueError(f'(FmowIWMDataset) keep_sensors: {set1} must be a subset of {set2}')
        
        if not set2.issubset(set1): #i.e. dont keep all sensors
            # self.df = self.df[self.df['sensor'].apply(lambda x: any([s in self.keep_sensors for s in x]))].reset_index(drop=True)
            def filter_columns_sensor(row):
                valid_indices = [i for i, sensor in enumerate(row['sensor']) if sensor in self.keep_sensors]
                
                for col in self.df.columns:
                    # if isinstance(row[col], list) or isinstance(row[col], np.ndarray):
                    if col in self.list_cols:
                        try:
                            row[col] = np.array(row[col])[valid_indices].tolist() if isinstance(row[col], list) else row[col][valid_indices]
                        except:
                            logger.info(f'Error in column: {col}, row: {row[col]}')
                return row
            self.df = self.df.apply(filter_columns_sensor, axis=1)
            logger.info(f'Subsetted dataset to only include rows where every row has atleast one of the following sensors: {self.keep_sensors} | #rows = {len(self.df)}')
            self.log_stats()

        
        #Subset the dataset to only include rows where the minimum image size >= self.max_crop
        if self.max_crop is not None:
            self.df = self.df[self.df['img_size'].apply(lambda x: any([sz >= self.max_crop for sz in x]))].reset_index(drop=True)
            logger.info(f'Subsetted dataset to only include rows where every row has atleast one image >= {self.max_crop} | #rows = {len(self.df)}')
            self.log_stats()

        if self.min_crop is not None:
            def filter_columns_img_size(row):
                valid_indices = [i for i, size in enumerate(row['img_size']) if size >= self.min_crop]
                
                for col in self.df.columns:
                    # if isinstance(row[col], list) or isinstance(row[col], np.ndarray):
                    if col in self.list_cols:
                        try:
                            row[col] = np.array(row[col])[valid_indices].tolist() if isinstance(row[col], list) else row[col][valid_indices]
                        except:
                            logger.info(f'Error in column: {col}, row: {row[col]}')
                return row

            # Apply the filter_columns function to each row
            self.df = self.df.apply(filter_columns_img_size, axis=1)
            logger.info(f'Subsetted dataset to only include rows where all image sizes >= {self.min_crop} | #rows = {len(self.df)}')
            self.log_stats()

        self.df['n'] = self.df['path'].str.len() #works with newer versions of pandas
        #check for NA values in the 'n'
        if self.df['n'].isna().sum() > 0:
            print(f'Warning: Found NA values in the column "n": removing {self.df["n"].isna().sum()} rows')
        
        self.df = self.df.dropna(subset=['n']).reset_index(drop=True)
        # self.df['uniq_sensors'] = self.df['sensor'].apply(lambda x: set(x))
        # self.df['nuniq_sensors'] = self.df['uniq_sensors'].apply(len)
        # Subset the dataset to only include rows where the number of available sensors >= self.M
        self.df = self.df[self.df['n'] >= self.M]
        logger.info(f'Subsetted dataset to only include rows where the number of available sensors >= {self.M} | #rows = {len(self.df)}')
        self.log_stats()

    def _load_img(self, path) -> torch.Tensor:
        path = os.path.join(self.root, path)
        return torch.from_numpy(imread(path).astype('float')).permute(2,0,1)


    def _format_timestamp(self, ts):
        if self.ref_date is None:
            #convert from object to numpy datetime64 to python datetime
            ts = ts.astype('M8[D]').astype('O')
            return torch.tensor([ts.year, ts.month, ts.day], dtype=torch.int16)

        else:
            ts = ts.astype('M8[D]').astype('O')        
            #convert ref_date from string to python datetime
            delta = (ts - self.ref_date).days
            return torch.tensor(delta, dtype=torch.long)


    def _format_gsd(self, gsd):
        return torch.tensor(gsd, dtype=torch.float32)


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # select one sensor whose image size is >= self.max_crop
        # This ensures that atleast one sensor is the full grid size
        large_enough_idx = [i for i, sz in enumerate(row['img_size']) if sz >= self.max_crop]


        def _get_idx():
            valid_sensors = list(range(len(row['img_size'])))
            if len(valid_sensors) < (self.M - 1):
                raise ValueError('Not enough sensors')  # Not enough unique sensors
            
            idx = np.random.choice(large_enough_idx, 1)[0]
            valid_sensors.remove(idx)
            idxs = np.random.choice(valid_sensors, self.M - 1, replace=False)
            idxs = np.append(idxs, idx)
            return idxs

        # load images with retries if failed
        tries = 0
        while tries < self.max_tries_load_img:
            idxs = _get_idx()
            imgs = []
            for i in idxs:
                try: 
                    imgs.append(self._load_img(row['path'][i]))
                except Exception as e:
                    faulty_path = os.path.join(self.root, row['path'][i])
                    full_traceback = traceback.format_exc()

                    logger.info(f'Error loading image: {faulty_path}')
                    if self.faulty_imgs_file is not None:
                        with open(self.faulty_imgs_file, 'a') as f:
                            f.write('\n\n')
                            f.write('time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
                            f.write('file: ' + faulty_path + '\n')
                            f.write(full_traceback)
                    break
            if len(imgs) == len(idxs):
                break
            tries += 1

        if tries == self.max_tries_load_img:
            raise ValueError(f'Failed to load images in {self.max_tries_load_img} tries')

        # load metainfo
        gsds = []
        times = []
        ds_names = []
        for i in idxs:
            ds_names.append(self.sensor_name_mapping[row['sensor'][i]])
            gsds.append(self._format_gsd(row['gsd'][i]))
            times.append(self._format_timestamp(row['timestamp'][i]))
        gsds = torch.stack(gsds, dim=0)  # M x ...
        times = torch.stack(times, dim=0) # M x ...
        chn_ids = [self.chn_ids[ds] for ds in ds_names]


        if self.normalize:
            for i in idxs:
                img = imgs.pop(0)
                sensor = row['sensor'][i]
                sensor = self.sensor_name_mapping[sensor]
                imgs.append(self.channelwise_transforms[sensor](img))

        if self.return_rgb:
            for i in idxs:
                img = imgs.pop(0)
                chn_id = chn_ids.pop(0)
                sensor = row['sensor'][i]
                sensor = self.sensor_name_mapping[sensor]
                rgb_idx = torch.tensor(self.RGB_CHANNELS[sensor])
                imgs.append(img[rgb_idx])
                chn_ids.append(chn_id[rgb_idx])

        out = [dict(
            imgs = [imgs[i]],
            chn_ids = [chn_ids[i]],
        ) for i in range(self.M)]

        # out = dict(
        #     imgs = [[i] for i in imgs], 
        #     chn_ids = chn_ids,
        #     # gsds = gsds, 
        #     # times = times)  # innermost elmnt in imgs is c x h x w (with c,h,w potentially different)
        # )

        if self.transform:
            out = self.transform(out)
        return out
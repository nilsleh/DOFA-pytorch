import os

import numpy as np
import rasterio
import torch
from torchgeo.datasets.utils import percentile_normalization
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import glob
from torch.utils.data import random_split

import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
import torch


class FLAIR(torch.utils.data.Dataset):
    """Implementation of FLAIR Aerial image dataset.

    Image size: 512x512
    Image bands: R,G,B,NIR,nDSM
    """

    classes = [
        "background",
        "building",
        "pervious surface",
        "impervious surface",
        "bare soil",
        "water",
        "coniferous",
        "deciduous",
        "vineyard",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "swimming_pool",
        "snow",
        "clear cut",
        "mixed",
        "ligneous",
        "greenhouse",
        "other",
    ]

    dir_names = {
        "train": {
            "images": "flair_aerial_train",
            "masks": "flair_labels_train",
        },
        "test": {
            "images": "flair_2_aerial_test",
            "masks": "flair_2_labels_test",
        },
    }
    globs = {
        "images": "IMG_*.tif",
        "masks": "MSK_*.tif",
    }
    splits = ("train", "test")

    def __init__(self, root, split="train", bands="rgb", transforms=None):
        assert split in self.splits
        assert bands in ("rgb", "all")
        self.root = root
        self.transforms = transforms
        self.bands = bands
        self.split = split
        self.samples = self._load_files()

    def _load_files(self):
        """Return the paths of the files in the dataset.
        Args:
            root: root dir of dataset
        Returns:
            list of dicts containing paths for each pair of image, masks
        """
        images = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]["images"],
                    "**",
                    self.globs["images"],
                ),
                recursive=True,
            )
        )

        masks = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]["masks"],
                    "**",
                    self.globs["masks"],
                ),
                recursive=True,
            )
        )

        files = [dict(image=image, mask=mask) for image, mask in zip(images, masks)]

        return files

    def load_image(self, path):
        indices = (1, 2, 3) if self.bands == "rgb" else (1, 2, 3, 4, 5)
        with rasterio.open(path) as f:
            x = f.read(indices)
        x = torch.from_numpy(x).to(torch.float32)
        return x

    def load_mask(self, path):
        with rasterio.open(path) as f:
            x = f.read(1)
        # TODO replace values > 13 with 13 as "other" class
        x[x > 13] = 13
        # shift the classes to start from 0
        x -= 1
        x = torch.from_numpy(x).to(torch.long)
        return x

    def __getitem__(self, index):
        path = self.samples[index]["image"]
        image = self.load_image(path)

        if self.bands == "all":
            path = self.samples[index]["nir"]
            nir = self.load_image(path)
            image = torch.cat([image, nir], dim=0)

        path = self.samples[index]["mask"]
        mask = self.load_mask(path)

        sample = dict(image=image, mask=mask)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"][:3].numpy(), 0, 3)
        image = percentile_normalization(image, lower=0, upper=100, axis=(0, 1))

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = sample["mask"].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title("Image")

        if show_mask:
            axs[1].imshow(mask, interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[1].set_title("Label")

        if show_predictions:
            axs[2].imshow(prediction, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        """Initialize the data augmentation pipeline for the segmentation task.
        
        Args:
            split (str): The split of the dataset. Either 'train' or 'test'.
            size (int): The size of the image.
        """
        super().__init__()


        mean = torch.tensor([105.08, 110.87, 101.82])
        std = torch.tensor([52.17, 45.38, 44])

        if split == "train":
            self.transform = AugmentationSequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["image", "mask"],
            )
        else:
            self.transform = AugmentationSequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                data_keys=["image", "mask"],
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        aug_sample = self.transform(sample)
        # Kornia adds an additional batch dimension
        return aug_sample["image"].squeeze(0), aug_sample["mask"].squeeze(0).long()


class Flair2Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

        self.val_split_pct = 0.1

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size)

        dataset_train = FLAIR(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_test = FLAIR(
            root=self.root_dir, split="test", transforms=eval_transform
        )
        # split val into val and test
        generator = torch.Generator().manual_seed(0)
        dataset_train, dataset_val = random_split(dataset_train, [1 - self.val_split_pct, self.val_split_pct], generator=generator)
        
        return dataset_train, dataset_val, dataset_test
"""Fields of The World Dataset Wrapper."""

import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
import torch
from torchgeo.datasets import FieldsOfTheWorld
from torch.utils.data import random_split


class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        """Initialize the data augmentation pipeline for the segmentation task.

        Args:
            split (str): The split of the dataset. Either 'train' or 'test'.
            size (int): The size of the image.
            num_channels (int): The desired number of input channels for the model.
        """
        super().__init__()

        mean = torch.Tensor([0])
        std = torch.Tensor([1])

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

        return aug_sample["image"].squeeze(0), aug_sample["mask"].squeeze(0).long()


class FieldsOfTheWorldDataset(FieldsOfTheWorld):
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size)

        train_countries = ["austria"]

        val_countries = ["belgium"]

        test_countries = ["czechia"]

        dataset_train = FieldsOfTheWorld(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = FieldsOfTheWorld(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = FieldsOfTheWorld(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test

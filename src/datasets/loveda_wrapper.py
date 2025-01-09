import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
import torch
from torchgeo.datasets import LoveDA
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


class LoveDADataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

        self.test_split_pct = 0.1

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size)

        dataset_train = LoveDA(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = LoveDA(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        # split val into val and test
        generator = torch.Generator().manual_seed(0)
        dataset_val, dataset_test = random_split(dataset_val, [1 - self.test_split_pct, self.test_split_pct], generator=generator)
        
        return dataset_train, dataset_val, dataset_test

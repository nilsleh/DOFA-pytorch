import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
import torch
from torchgeo.datasets import CaFFe

class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size, num_channels):
        """Initialize the data augmentation pipeline for the segmentation task.
        
        Args:
            split (str): The split of the dataset. Either 'train' or 'test'.
            size (int): The size of the image.
            num_channels (int): The desired number of input channels for the model.
        """
        super().__init__()

        self.num_channels = num_channels

        mean = torch.Tensor([0.5517])
        std = torch.Tensor([11.8478])

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
        sample["mask"] = sample["mask_zones"]
        del sample["mask_zones"]
        aug_sample = self.transform(sample)
        if self.num_channels != 1:
            aug_sample["image"] = aug_sample["image"].expand(-1, self.num_channels, -1, -1)

        # TODO find the correct wavelength depending on the sample path
        return aug_sample["image"].squeeze(0), aug_sample["mask"].squeeze(0).long()


class CaffeDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.num_channels = config.num_channels

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size, num_channels=self.num_channels)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size, num_channels=self.num_channels)

        dataset_train = CaFFe(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = CaFFe(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = CaFFe(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test


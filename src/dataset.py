
import numpy as np
import os
from PIL import Image
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Optional, Callable


class ProjectDatasets(Dataset):
    """Class utility to load, pre-process, put in batch, and convert to PyTorch convention images from the TID2013 dataset.
    """
    root_path = "/net/ens/am4ip/datasets/project-dataset/"

    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """ Class initialization.

        :param transform: A set of transformations to apply on data.
        :param target_transform: A set of transformations to apply on labels.
        """
        # Combine rainy and sunny images and masks into a single list
        self.image_paths = []
        self.mask_paths = []
        self.source = []

        # Add rainy images and masks
        rainy_images = os.listdir(os.path.join(self.root_path, "rainy_images"))
        rainy_masks = os.listdir(os.path.join(self.root_path, "rainy_sseg"))
        for img, mask in zip(rainy_images, rainy_masks):
            self.image_paths.append(os.path.join(self.root_path, "rainy_images", img))
            self.mask_paths.append(os.path.join(self.root_path, "rainy_sseg", mask))
            self.source.append("rainy")

        # Add sunny images and masks
        sunny_images = os.listdir(os.path.join(self.root_path, "sunny_images"))
        sunny_masks = os.listdir(os.path.join(self.root_path, "sunny_sseg"))
        for img, mask in zip(sunny_images, sunny_masks):
            self.image_paths.append(os.path.join(self.root_path, "sunny_images", img))
            self.mask_paths.append(os.path.join(self.root_path, "sunny_sseg", mask))
            self.source.append("sunny")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Dataset size.
        :return: Size of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Retrieve image-mask pair for a given index.

        :param index: Index of the item to retrieve.
        :return: A dictionary containing an image and its corresponding mask.
        """
        # Validate the index
        if index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for the dataset of size {len(self)}.")

        # Retrieve image and mask paths
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        source = self.source[index]

        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        # Return the image and mask pair
        return image ,mask, source
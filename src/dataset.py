
import numpy as np
import os
from PIL import Image
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Optional, Callable


class ProjectDatasets(Dataset):
    """Class utility to load, pre-process, put in batch, and convert to PyTorch convention images from the TID2013 dataset.
    """
    # root_path = "C:\\RESOURCES\\datasets\\tid2013"
    root_path = "/net/ens/am4ip/datasets/project-dataset/"

    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """ Class initialization.

        :param transform: A set of transformations to apply on data.
        :param target_transform: A set of transformations to apply on labels.
        """
        # Get all rainy images
        self.rainy_images = os.listdir(os.path.join(self.root_path, "rainy_images/"))
        self.rainy_image_paths = np.array(self.rainy_images)
          
        # Get all groundtruth images images
        self.rainy_masks = os.listdir(os.path.join(self.root_path, "rainy_sseg/"))
        self.rainy_masks_paths = np.array(self.rainy_masks)

        self.sunny_images = os.listdir(os.path.join(self.root_path, "sunny_images/"))
        self.sunny_image_paths = np.array(self.sunny_images)

        self.sunny_masks = os.listdir(os.path.join(self.root_path, "sunny_sseg/"))
        self.sunny_mask_paths = np.array(self.sunny_masks)

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        """Dataset size.
        :return: Size of the dataset.
        """
        return len(self.rainy_image_paths) + len(self.sunny_image_paths)
    

    def __getitem__(self, index: int):
        """Retrieve sunny and rainy image-mask pairs for a given index.

        :param index: Index of the item to retrieve.
        :return: A dictionary containing sunny and rainy image-mask pairs, or an error message if the index is out of bounds.
        """
        # Validate the index
        if index >= len(self.rainy_image_paths):
            raise IndexError(f"Index {index} is out of bounds for the dataset of size {len(self.rainy_image_paths) }.")
        
        elif index >= len(self.sunny_image_paths):
            raise IndexError(f"Index {index} is out of bounds for the dataset of size {len(self.sunny_image_paths)}.")

        # Retrieve rainy image and mask
        rainy_image_path = os.path.join(self.root_path, "rainy_images", self.rainy_image_paths[index])
        rainy_mask_path = os.path.join(self.root_path, "rainy_sseg", self.rainy_masks_paths[index])
        rainy_image = Image.open(rainy_image_path)
        rainy_mask = Image.open(rainy_mask_path)
        
        # Retrieve sunny image and mask
        sunny_image_path = os.path.join(self.root_path, "sunny_images", self.sunny_image_paths[index])
        sunny_mask_path = os.path.join(self.root_path, "sunny_sseg", self.sunny_mask_paths[index])
        sunny_image = Image.open(sunny_image_path)
        sunny_mask = Image.open(sunny_mask_path)

        # Apply transformations if provided
        if self.transform:
            rainy_image = self.transform(rainy_image)
            sunny_image = self.transform(sunny_image)

        if self.target_transform:
            rainy_mask = self.target_transform(rainy_mask)
            sunny_mask = self.target_transform(sunny_mask)

        # Return both sunny and rainy pairs
        return {
            "rainy": (rainy_image, rainy_mask),
            "sunny": (sunny_image, sunny_mask),
        }

   
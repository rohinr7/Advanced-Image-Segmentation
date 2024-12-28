import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms.functional import to_tensor
import numpy as np


def mask_to_class_index(mask):
    """Converts an RGB mask to a class index mask using vectorized operations."""
    mask = np.array(mask)
    class_indices = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Define color-to-class mapping
    color_to_class = {
        (0, 0, 0): 0,            # Unlabeled
        (111, 74, 0): 1,         # Dynamic
        (81, 0, 81): 2,          # Ground
        (128, 64, 128): 3,       # Road
        (244, 35, 232): 4,       # Sidewalk
        (250, 170, 160): 5,      # Parking
        (230, 150, 140): 6,     # Rail track
        (70, 70, 70): 7,        # Building
        (102, 102, 156): 8,     # Wall
        (190, 153, 153): 9,     # Fence
        (180, 165, 180): 10,     # Guard rail
        (150, 100, 100): 11,     # Bridge
        (150, 120, 90): 12,      # Tunnel
        (153, 153, 153): 13,     # Pole
        (250, 170, 30): 14,      # Traffic light
        (220, 220, 0): 15,       # Traffic sign
        (107, 142, 35): 16,      # Vegetation
        (152, 251, 152): 17,     # Terrain
        (70, 130, 180): 18,      # Sky
        (220, 20, 60): 19,       # Person
        (255, 0, 0): 20,         # Rider
        (0, 0, 142): 21,         # Car
        (0, 0, 70): 22,          # Truck
        (0, 60, 100): 23,        # Bus
        (0, 0, 90): 24,          # Caravan
        (0, 0, 110): 25,         # Trailer
        (0, 80, 100): 26,        # Train
        (0, 0, 230): 27,         # Motorcycle
        (119, 11, 32): 29        # Bicycle
    }

    # Convert color-to-class mapping into numpy arrays
    colors = np.array(list(color_to_class.keys()), dtype=np.uint8)
    class_ids = np.array(list(color_to_class.values()), dtype=np.uint8)

    # Reshape for broadcasting
    reshaped_mask = mask.reshape(-1, 3)  # Flatten mask to [num_pixels, 3]
    reshaped_colors = colors.reshape(1, -1, 3)  # Shape [1, num_classes, 3]

    # Compare all pixels with all colors
    matches = np.all(reshaped_mask[:, None, :] == reshaped_colors, axis=2)

    # Assign class indices based on matches
    class_indices = class_ids[np.argmax(matches, axis=1)].reshape(mask.shape[:2])

    return class_indices



from torchvision.transforms import Resize

class ProjectDatasets(Dataset):
    def __init__(self, root_path, transform=None, target_transform=None):
        self.datasets = []
        self.sources = []
        self.transform = transform
        self.target_transform = target_transform
        self.resize_transform = Resize((256, 256))  # Example size

        for subset in ["rainy", "sunny"]:
            image_dir = os.path.join(root_path, f"{subset}_images")
            mask_dir = os.path.join(root_path, f"{subset}_sseg")

            image_paths = sorted(os.listdir(image_dir))
            mask_paths = sorted(os.listdir(mask_dir))

            for img, msk in zip(image_paths, mask_paths):
                self.datasets.append((os.path.join(image_dir, img), os.path.join(mask_dir, msk)))
                self.sources.append(subset)

    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        image_path, mask_path = self.datasets[index]
        source = self.sources[index]

        # Load and resize image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image = self.resize_transform(image)
        mask = self.resize_transform(mask)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = to_tensor(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = mask_to_class_index(mask)
            mask = torch.from_numpy(mask.astype(np.int64))
        image_name = os.path.basename(image_path)
        return image, mask, source ,image_name

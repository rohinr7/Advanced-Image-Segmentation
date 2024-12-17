from torch.utils.data import Dataset
from PIL import Image
import os

class ProjectDatasets(Dataset):
    """
    A dataset class that combines rainy and sunny datasets and tracks the source of each sample.
    """
    def __init__(self, root_path, transform=None, target_transform=None):
        self.datasets = []
        self.sources = []
        self.transform = transform
        self.target_transform = target_transform

        for subset in ["rainy", "sunny"]:
            image_dir = os.path.join(root_path, f"{subset}_images")
            mask_dir = os.path.join(root_path, f"{subset}_sseg")

            image_paths = sorted(os.listdir(image_dir))
            mask_paths = sorted(os.listdir(mask_dir))

            for img, msk in zip(image_paths, mask_paths):
                self.datasets.append((os.path.join(image_dir, img), os.path.join(mask_dir, msk)))
                self.sources.append(subset)  # Track dataset source

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image_path, mask_path = self.datasets[index]
        source = self.sources[index]  # Source dataset (rainy or sunny)

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = mask_to_class_index(mask)  # Default mapping to class indices

        return image, mask, source

# src/utils/transform_utils.py
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter

def get_transforms(augmentation_config):
    """
    Create a Compose object with optional data augmentations based on the config.

    Args:
        augmentation_config (dict): Configuration for augmentations.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    transforms = [ToTensor()]  # Base transformation to convert PIL to Tensor

    if augmentation_config.get("use_augmentation", False):
        aug_params = augmentation_config.get("params", {})
        if aug_params.get("horizontal_flip", False):
            transforms.append(RandomHorizontalFlip())  # Apply horizontal flip
        if aug_params.get("rotation_range", 0) > 0:
            transforms.append(RandomRotation(aug_params["rotation_range"]))  # Apply random rotation
        if aug_params.get("brightness_adjust", 0) > 0:
            transforms.append(ColorJitter(brightness=aug_params["brightness_adjust"]))  # Adjust brightness

    # Always add normalization at the end
    transforms.append(Normalize(mean=[0.5], std=[0.5]))

    return Compose(transforms)

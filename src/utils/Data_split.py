import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Function to split dataset into training, validation, and test sets
def split_dataset(dataset, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
    """Splits the dataset into training, validation, and test sets.

    :param dataset: The dataset to split.
    :param train_ratio: Ratio of the dataset to be used for training.
    :param val_ratio: Ratio of the dataset to be used for validation.
    :param test_ratio: Ratio of the dataset to be used for testing.
    :param seed: Random seed for reproducibility.
    :return: A tuple of (train_dataset, val_dataset, test_dataset).
    """
    
    if not (train_ratio + val_ratio + test_ratio) < 1:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    torch.manual_seed(seed)  # Set random seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

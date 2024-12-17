import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ProjectDatasets
from utils.Data_split import split_dataset 

# Transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a consistent size
    transforms.ToTensor(),          # Convert to Tensor
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = ProjectDatasets(
    transform=image_transform,
    target_transform=mask_transform
)

# Split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = split_dataset(dataset)

# Create DataLoaders for each split
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


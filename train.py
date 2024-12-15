import torch
from torchvision.transforms import Compose, PILToTensor
from dataset import ProjectDatasets
import matplotlib.pyplot as plt
from torchvision import transforms
import random

# Instantiate the dataset
train_dataset = ProjectDatasets(transform=None)

# Number of samples to display
num_samples = 3

# Create a subplot grid (5 rows x 4 columns: rainy image, rainy mask, sunny image, sunny mask)
fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))

# Randomly select indices
random_indices = random.sample(range(int(len(train_dataset)/2)), num_samples)

for i, idx in enumerate(random_indices):
    # Fetch data for the random index
    data = train_dataset[idx]
    rainy_image, rainy_mask = data["rainy"]
    sunny_image, sunny_mask = data["sunny"]

    # Plot rainy image
    axes[i, 0].imshow(rainy_image)
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"Rainy Image {idx}")

    # Plot rainy mask
    axes[i, 1].imshow(rainy_mask)
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f"Rainy Mask {idx}")

    # Plot sunny image
    axes[i, 2].imshow(sunny_image)
    axes[i, 2].axis('off')
    axes[i, 2].set_title(f"Sunny Image {idx}")

    # Plot sunny mask
    axes[i, 3].imshow(sunny_mask)
    axes[i, 3].axis('off')
    axes[i, 3].set_title(f"Sunny Mask {idx}")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
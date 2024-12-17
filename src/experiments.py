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
fig, axes = plt.subplots(num_samples, 2, figsize=(16, num_samples * 4))

# Randomly select indices
random_indices = random.sample(range(len(train_dataset)), num_samples)

for i, idx in enumerate(random_indices):
    # Fetch data for the random index
    data = train_dataset[idx]
    dat_image, img_mask , source = data
    

    # Plot rainy image
    axes[i, 0].imshow(dat_image)
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"{source} Image {idx}")

    # Plot rainy mask
    axes[i, 1].imshow(img_mask)
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f"{source}Mask {idx}")


# Adjust layout and display the plot
plt.tight_layout()
plt.show()
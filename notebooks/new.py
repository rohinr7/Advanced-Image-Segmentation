import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
from src.dataset import ProjectDatasets
from src.models.UNet import UNet  # Replace with your model architecture
import random
import torch
import numpy as np
class DataVisualizer:
    def __init__(self, dataset, model, device, batch_size=1):
        """
        Initialize the Data Visualizer with dataset, model, and batch size.
        """
        self.dataset = dataset
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # Calculate split sizes
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size  # Remaining 20% for validation

        # Reproducible split
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        self.loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.data = list(self.loader)  # Load all data for visualization
        self.num_samples = len(self.data)
        self.current_index = 0
        self.model = model
        self.device = device
        print(f"Loaded {self.num_samples} samples from dataset.")

    def run_model(self, image):
        """
        Run the model on the given image and return the prediction.
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image.unsqueeze(0))  # Add batch dimension
            prediction = output.argmax(dim=1).squeeze(0)  # Get predicted class
        return prediction.cpu()

    def display_sample(self, index):
        """
        Display the original image, ground truth, and model prediction for a given index.
        """
        if index < 0 or index >= self.num_samples:
            print("Index out of range!")
            return
        
        # Retrieve the image and ground truth
        image, mask = self.data[index]

        # Run the model on the image
        prediction = self.run_model(image)

        # Prepare the figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Display the original image
        axes[0].imshow(image.squeeze(0).permute(1, 2, 0).numpy(), cmap="gray")  # Adjust channel order
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Display the ground truth
        axes[1].imshow(mask.squeeze(0).numpy(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Display the model prediction
        axes[2].imshow(prediction.numpy(), cmap="gray")
        axes[2].set_title("Model Prediction")
        axes[2].axis("off")

        plt.show()

    def start_visualization(self):
        """
        Start the interactive visualization using keyboard input.
        """
        paused = False

        while True:
            if not paused:
                print(f"Displaying sample {self.current_index + 1}/{self.num_samples}")
                self.display_sample(self.current_index)

            # Keyboard commands
            print("Commands: [n] Next | [p] Previous | [q] Quit | [pause] Pause/Play")
            command = input("Enter command: ").strip().lower()

            if command == "n":
                self.current_index = (self.current_index + 1) % self.num_samples  # Move to the next sample
                paused = False
            elif command == "p":
                self.current_index = (self.current_index - 1) % self.num_samples  # Move to the previous sample
                paused = False
            elif command == "pause":
                paused = not paused  # Toggle pause/play
                print("Paused" if paused else "Playing")
            elif command == "q":
                print("Exiting visualization.")
                break
            else:
                print("Invalid command. Please enter [n], [p], [pause], or [q].")

def main(data_path, checkpoint_path, batch_size=1):
    """
    Main function to load the dataset, model, and start the visualizer.
    """
    # Define transformations for the dataset
    transform = Compose([
        ToTensor(),  # Convert images to PyTorch tensors
        Normalize(mean=[0.5], std=[0.5]),  # Normalize the dataset
    ])

    # Load the dataset
    dataset = ProjectDatasets(root_path=data_path, transform=transform)

    # Initialize the model
    model = UNet(in_channels=3, out_channels=30)  # Replace with your model and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the model checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully.")

    # Initialize the visualizer
    visualizer = DataVisualizer(dataset, model, device, batch_size=batch_size)
    print("here we are after loading visulizer")

    # Start the visualization
    visualizer.start_visualization()
    print("we hare now")


if __name__ == "__main__":
    # Parameters (update paths as needed)
    DATA_PATH = "/net/ens/am4ip/datasets/project-dataset"  # Path to dataset
    CHECKPOINT_PATH = "/net/cremi/sasifchaudhr/espaces/travail/Semantic-Segmentation-for-Autonomous-Driving/experiments/experiment_20241220-131602/checkpoints/checkpoint_epoch_10.pth"  # Path to checkpoint

    # Run the visualizer
    main(DATA_PATH, CHECKPOINT_PATH)
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
from src.dataset import ProjectDatasets
from src.models.UNet import UNet
import random
import numpy as np
import time
import threading
from queue import Queue


class DataVisualizer:
    def __init__(self, dataset, model, device, batch_size=1, preload_size=10):
        self.dataset = dataset
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(seed)
        _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        # DataLoader without preloading
        self.loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.num_samples = len(val_dataset)
        self.current_index = 0
        self.model = model
        self.device = device
        self.playing = True  # Start in play mode
        self.stop_thread = False  # Control thread exit
        self.command = None  # Store the latest command

        # Preloading queue
        self.preload_queue = Queue(maxsize=preload_size)
        self.preloader_thread = threading.Thread(target=self._preload_data)
        self.preloader_thread.daemon = True
        self.preloader_thread.start()

        # Input listener thread
        self.input_thread = threading.Thread(target=self._listen_for_commands)
        self.input_thread.daemon = True
        self.input_thread.start()

        print(f"Loaded {self.num_samples} samples from dataset.")

    def _preload_data(self):
        """
        Preload data (image, ground truth, prediction) into a queue in the background.
        """
        while not self.stop_thread:
            if not self.preload_queue.full():
                image, mask, _ = self.loader.dataset[self.current_index]
                prediction = self.run_model(image)
                self.preload_queue.put((image, mask, prediction))
                self.current_index = (self.current_index + 1) % self.num_samples
            else:
                time.sleep(0.1)  # Avoid busy-waiting

    def _listen_for_commands(self):
        """
        Listen for user commands in the background.
        """
        while not self.stop_thread:
            command = input().strip().lower()  # Wait for user input
            self.command = command  # Store the command for processing

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

    def display_sample(self):
        """
        Display the next sample from the preload queue.
        """
        if not self.preload_queue.empty():
            image, mask, prediction = self.preload_queue.get()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image.permute(1, 2, 0).numpy())
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(mask.numpy(), cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(prediction.numpy(), cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

        else:
            print("Waiting for data to be preloaded...")

    def start_visualization(self):
        """
        Start the visualization with continuous playback unless interrupted by commands.
        """
        while not self.stop_thread:
            if self.playing:
                print(f"Displaying sample {self.current_index + 1}/{self.num_samples}")
                self.display_sample()
                time.sleep(1)  # Pause for 1 second between frames

            # Process the latest command, if any
            if self.command:
                if self.command == "pause":
                    print("Paused.")
                    self.playing = False
                elif self.command == "play":
                    print("Resuming playback.")
                    self.playing = True
                elif self.command == "rewind":
                    print("Rewinding to the start.")
                    self.current_index = 0
                elif self.command == "n":
                    print("Moving to the next frame.")
                    self.current_index = (self.current_index + 1) % self.num_samples
                elif self.command == "p":
                    print("Moving to the previous frame.")
                    self.current_index = (self.current_index - 1) % self.num_samples
                elif self.command == "q":
                    print("Exiting visualization.")
                    self.stop_thread = True
                    self.preloader_thread.join()  # Ensure preloader thread exits
                    break
                else:
                    print(f"Unknown command: {self.command}")
                self.command = None  # Reset the command after processing


def main(data_path, checkpoint_path, batch_size=1):
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = ProjectDatasets(root_path=data_path, transform=transform)

    model = UNet(in_channels=3, out_channels=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Pre-warm CUDA
    if torch.cuda.is_available():
        _ = torch.randn(1).to(device)

    # Load checkpoint (local copy)
    local_checkpoint = "/tmp/checkpoint_epoch_10.pth"
    os.system(f"cp {checkpoint_path} {local_checkpoint}")
    checkpoint = torch.load(local_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    visualizer = DataVisualizer(dataset, model, device, batch_size=batch_size)
    visualizer.start_visualization()


if __name__ == "__main__":
    # Parameters (update paths as needed)
    DATA_PATH = "/net/ens/am4ip/datasets/project-dataset"
    CHECKPOINT_PATH = "/net/cremi/sasifchaudhr/espaces/travail/Semantic-Segmentation-for-Autonomous-Driving/experiments/experiment_20241220-131602/checkpoints/checkpoint_epoch_10.pth"

    main(DATA_PATH, CHECKPOINT_PATH)
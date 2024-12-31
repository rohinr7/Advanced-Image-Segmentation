import torch
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Reverse the color_to_class mapping to get class_to_color mapping
color_to_class = {
    (0, 0, 0): 0,            # Unlabeled
    (111, 74, 0): 1,         # Dynamic
    (81, 0, 81): 2,          # Ground
    (128, 64, 128): 3,       # Road
    (244, 35, 232): 4,       # Sidewalk
    (250, 170, 160): 5,      # Parking
    (230, 150, 140): 6,      # Rail track
    (70, 70, 70): 7,         # Building
    (102, 102, 156): 8,      # Wall
    (190, 153, 153): 9,      # Fence
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

class_to_color = {v: k for k, v in color_to_class.items()}

def map_classes_to_colors(predictions, class_to_color):
    """
    Map class IDs to RGB colors for a segmentation map.

    Args:
        predictions (torch.Tensor): Tensor of shape (H, W) containing class IDs.
        class_to_color (dict): Mapping from class IDs to RGB colors.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
    """
    height, width = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in class_to_color.items():
        rgb_image[predictions == class_id] = color  # Map each class to its color
    
    return rgb_image



def save_checkpoint(model, optimizer, epoch, experiment_dir, is_best=False):
    """
    Save model checkpoints.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        experiment_dir (str): Directory to save checkpoints.
        is_best (bool): If True, save as the best checkpoint.
    """
    # Save the latest checkpoint (overwrites previous last checkpoint)
    last_checkpoint_path = os.path.join(experiment_dir, "last_checkpoint.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, last_checkpoint_path)
    print(f"Saved last checkpoint to {last_checkpoint_path}")

    # Save the best checkpoint if required
    if is_best:
        best_path = os.path.join(experiment_dir, "best_checkpoint.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, best_path)
        print(f"Saved best checkpoint to {best_path}")
        

def load_checkpoint(filepath, model, optimizer=None):
    """
    Load the model and optimizer state from a checkpoint file.
    
    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into (if available).

    Returns:
        model (torch.nn.Module): Model with loaded state.
        optimizer (torch.optim.Optimizer): Optimizer with loaded state (if provided).
        epoch (int): The epoch number from the checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {filepath}, resuming at epoch {epoch}")
    return model, optimizer, epoch

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Ensures all tensors in the batch have consistent shapes and types.
    """
    images, masks, sources = zip(*batch)  # Unpack the batch

    # Stack images and masks into tensors
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return images, masks, sources

def save_config(config, experiment_dir):
    """Save experiment configuration to a JSON file."""
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def log_metrics(epoch, train_loss, val_loss, train_accuracy, val_metrics, experiment_dir, class_names=None):
    """
    Logs training and validation losses, accuracy, and metrics to a file.

    Args:
        epoch (int): Current epoch.
        train_loss (float): Training loss for the epoch.
        val_loss (float): Validation loss for the epoch.
        train_accuracy (float): Training accuracy for the epoch.
        val_metrics (dict): Validation metrics (e.g., IoU, Pixel Accuracy, DICE).
        experiment_dir (str): Path to the experiment directory.
        class_names (list): Optional list of class names for better readability.
    """
    log_path = os.path.join(experiment_dir, "train.log")

    # Prepare class names if not provided
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(val_metrics["IoU"]))]

    # Format class-wise metrics for IoU and DICE
    formatted_iou = ", ".join(f"{class_names[i]}: {val_metrics['IoU'][i]:.4f}" for i in range(len(class_names)))
    formatted_dice = ", ".join(f"{class_names[i]}: {val_metrics['DICE'][i]:.4f}" for i in range(len(class_names)))

    # Format overall metrics
    metrics_str = f"""
    Epoch {epoch}:
      Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}
      Val Loss: {val_loss:.4f}
      Metrics:
        val_loss: {val_metrics['val_loss']:.4f}
        val_accuracy: {val_metrics['val_accuracy']:.4f}
        IoU: {{ {formatted_iou} }}
        MeanIoU: {val_metrics['MeanIoU']:.4f}
        PixelAccuracy: {val_metrics['PixelAccuracy']:.4f}
        DICE: {{ {formatted_dice} }}
        MeanDICE: {val_metrics['MeanDICE']:.4f}
    """

    # Write to log file
    with open(log_path, "a") as f:
        f.write(metrics_str + "\n")

    # Print to console for visibility
    print(metrics_str)



def save_checkpoint(model, optimizer, epoch, experiment_dir, is_best=False):
    """
    Saves a checkpoint of the model and optimizer state.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        experiment_dir (str): The experiment directory where the checkpoint will be saved.
        is_best (bool): If True, saves the checkpoint as 'best_model.pth'.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

    # Save as best model if is_best is True
    if is_best:
        best_model_path = os.path.join(experiment_dir, "checkpoints", "best_model.pth")
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved at: {best_model_path}")


def save_predictions(inputs, targets, predictions, epoch, experiment_dir):
    """Save example predictions for visualization."""
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Convert multi-channel predictions to single-channel by taking argmax
    predictions = predictions.argmax(axis=1)

    # Save the first few examples
    for i in range(min(len(inputs), 5)):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input image (rescale to [0, 1] for imshow compatibility)
        axs[0].imshow(inputs[i].transpose(1, 2, 0))
        axs[0].set_title("Input")
        
        # Ground truth mask
        axs[1].imshow(targets[i].squeeze(), cmap="gray")
        axs[1].set_title("Ground Truth")
        
        # Predicted mask
        axs[2].imshow(predictions[i], cmap="gray")
        axs[2].set_title("Prediction")
        
        for ax in axs:
            ax.axis("off")
        
        fig.savefig(os.path.join(experiment_dir, "results", f"epoch_{epoch}_sample_{i}.png"))
        plt.close(fig)    


def visualize_batch_with_colorbar(inputs, predictions, targets, batch_idx, num_samples=3, rgb_pred=False):
    """
    Visualize a few samples from the batch with intensity color bars and optional RGB predictions.
    
    Args:
        inputs (torch.Tensor): Input images of shape (N, C, H, W).
        predictions (torch.Tensor or np.ndarray): Model predictions of shape (N, H, W) or (N, H, W, 3).
        targets (torch.Tensor): Ground truth masks of shape (N, H, W).
        batch_idx (int): Batch index (for display purposes).
        num_samples (int): Number of samples to visualize.
        rgb_pred (bool): If True, predictions are RGB images.
    """
    inputs = inputs.cpu().numpy()
    #targets = targets.cpu().numpy()
    if rgb_pred == False:
        predictions_r = predictions.cpu().numpy()
    
    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input image
        ax1 = axes[0]
        im1 = ax1.imshow(inputs[i].transpose(1, 2, 0))  # Convert CHW to HWC for display
        ax1.set_title(f"Input Image (Batch {batch_idx}, Sample {i})")
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)  # Add color bar
        
        # Prediction
        ax2 = axes[1]
        if rgb_pred:
            im2 = ax2.imshow(predictions[i])  # RGB prediction
        else:
            im2 = ax2.imshow(predictions_r[i], cmap="gray")  # Grayscale/class prediction
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)  # Add color bar
        ax2.set_title(f"Prediction (Sample {i})")
        ax2.axis("off")
        
        # Ground truth
        ax3 = axes[2]
        im3 = ax3.imshow(targets[i])  # Use the same colormap

        ax3.set_title(f"Ground Truth (Sample {i})")
        ax3.axis("off")

        
        plt.tight_layout()
        plt.show()

import yaml

def load_config(config_path):
    print(f"Loading config from: {config_path}")  # Debug statement
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"Loaded config type: {type(config)}")  # Should be <class 'dict'>
        return config
    
def log_metrics_test(
    experiment_dir: str,
    metrics: dict, 
    class_names: list = None, 
    indent: int = 2, 
    decimals: int = 4
) -> None:
    """
    Print metrics in a nicely formatted way and also log them to a file. 
    If class_names is provided and the sub-key is an integer that corresponds
    to an index in class_names, print the class name alongside its index.

    :param experiment_dir: Path to the experiment directory where logs should be saved.
    :param metrics: Dictionary containing nested metrics.
    :param class_names: List or dict of class names. If list, the index must
                        match the sub-key. If dict, sub-key must exist as a key.
    :param indent: Number of spaces to use for indentation.
    :param decimals: Number of decimal places to show for float values.
    """
    # Determine the log file path and ensure the directory exists
    log_path = os.path.join(experiment_dir, "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    spacer = " " * indent

    # We'll accumulate all lines in a list, then join them into one string.
    output_lines = []
    output_lines.append("Results on Test Data Set :")

    for key, val in metrics.items():
        if isinstance(val, dict):
            # Nested metrics dictionary
            output_lines.append(f"{spacer}{key}:")
            for sub_key, sub_val in val.items():
                # Attempt to resolve class name if sub_key is an integer or if it exists in class_names
                class_label = None
                if class_names is not None:
                    if isinstance(class_names, list) and isinstance(sub_key, int) and sub_key < len(class_names):
                        class_label = class_names[sub_key]
                    elif isinstance(class_names, dict) and sub_key in class_names:
                        class_label = class_names[sub_key]

                # Format the output key with possible class name
                if class_label is not None:
                    label_str = f"{sub_key} ({class_label})"
                else:
                    label_str = str(sub_key)

                # Format float values
                if isinstance(sub_val, float):
                    output_lines.append(f"{spacer*2}{label_str}: {sub_val:.{decimals}f}")
                else:
                    output_lines.append(f"{spacer*2}{label_str}: {sub_val}")
        else:
            # Top-level metric
            if isinstance(val, float):
                output_lines.append(f"{spacer}{key}: {val:.{decimals}f}")
            else:
                output_lines.append(f"{spacer}{key}: {val}")

    # Create a single string from the lines
    final_output = "\n".join(output_lines)

    # Print to console
    print(final_output)

    # Append to the log file
    with open(log_path, "a") as f:
        f.write(final_output + "\n\n")

def log_to_file(message, experiment_folder, filename="train.log"):
    """
    Logs a message to a specified file within the experiment folder.

    Args:
        message (str): The message to log.
        experiment_folder (str): Path to the experiment folder.
        filename (str): Name of the file to log into (default: "log.txt").

    Returns:
        None
    """
    # Ensure the experiment folder exists
    os.makedirs(experiment_folder, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(experiment_folder, filename)

    # Append the message to the file
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
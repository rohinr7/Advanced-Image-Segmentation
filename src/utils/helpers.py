import torch
import os
import json
import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save the model and optimizer state to a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        filepath (str): Path to save the checkpoint file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


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

def log_metrics(epoch, train_loss, val_loss, experiment_dir):
    """Log training and validation losses to a file."""
    log_path = os.path.join(experiment_dir, "logs", "train.log")
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")

def save_checkpoint(model, optimizer, epoch, experiment_dir):
    """Save model and optimizer state."""
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

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
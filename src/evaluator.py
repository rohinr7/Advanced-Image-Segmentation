import torch
import numpy as np
from src.metrics import compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from src.utils.helpers import map_classes_to_colors
import os
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, device, class_to_color, metrics_config):
        """
        Initializes the evaluator.

        Args:
            model (torch.nn.Module): Trained model.
            device (torch.device): Device to run evaluation on.
            class_to_color (dict): Mapping of class IDs to RGB colors.
            metrics_config (dict): Configuration for metrics to evaluate (e.g., {'iou': True, 'pixel_accuracy': False}).
        """
        self.model = model
        self.device = device
        self.class_to_color = class_to_color
        self.num_classes = len(class_to_color)
        self.metrics_config = metrics_config

    def evaluate_batch(self, predictions, targets):
        """
        Compute metrics for a single batch.

        Args:
            predictions (np.ndarray): Predicted class IDs of shape (N, H, W).
            targets (np.ndarray): Ground truth class IDs of shape (N, H, W).

        Returns:
            dict: Dictionary containing computed metrics based on the enabled configuration.
        """
        results = {}
        if self.metrics_config.get("iou", False):
            results["IoU"], results["MeanIoU"] = compute_iou(predictions, targets, self.num_classes)
        if self.metrics_config.get("pixel_accuracy", False):
            results["PixelAccuracy"] = compute_pixel_accuracy(predictions, targets)
        if self.metrics_config.get("dice", False):
            results["DICE"], results["MeanDICE"] = compute_dice_coefficient(predictions, targets, self.num_classes)
        return results

    def evaluate(self, data_loader, save_rgb=False, output_dir=None):
        """
        Evaluate the model on the given data loader.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation dataset.
            save_rgb (bool): Whether to save RGB visualizations of predictions and targets.
            output_dir (str): Directory to save RGB visualizations if `save_rgb` is True.

        Returns:
            dict: Aggregated metrics across all batches.
        """
        print("Starting evaluation loop...")
        self.model.eval()

        # Initialize total metrics based on enabled configuration
        total_metrics = {}
        if self.metrics_config.get("iou", False):
            total_metrics["IoU"] = []
            total_metrics["MeanIoU"] = []
        if self.metrics_config.get("pixel_accuracy", False):
            total_metrics["PixelAccuracy"] = []
        if self.metrics_config.get("dice", False):
            total_metrics["DICE"] = []
            total_metrics["MeanDICE"] = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, _, _) in enumerate(data_loader):
                print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs)
                predictions = torch.argmax(predictions, dim=1).cpu().numpy()
                targets = targets.cpu().numpy()

                # Compute batch metrics
                batch_metrics = self.evaluate_batch(predictions, targets)
                print(f"Batch {batch_idx + 1} Metrics: {batch_metrics}")

                # Aggregate metrics
                for metric, value in batch_metrics.items():
                    total_metrics[metric].append(value)

                # Optionally save RGB visualizations
                if save_rgb and output_dir:
                    rgb_predictions = np.stack([
                        map_classes_to_colors(predictions[i], self.class_to_color) for i in range(predictions.shape[0])
                    ])
                    rgb_targets = np.stack([
                        map_classes_to_colors(targets[i], self.class_to_color) for i in range(targets.shape[0])
                    ])
                    for i, (rgb_pred, rgb_target) in enumerate(zip(rgb_predictions, rgb_targets)):
                        pred_path = os.path.join(output_dir, f"batch_{batch_idx + 1}_sample_{i}_pred.png")
                        target_path = os.path.join(output_dir, f"batch_{batch_idx + 1}_sample_{i}_target.png")
                        plt.imsave(pred_path, rgb_pred)
                        plt.imsave(target_path, rgb_target)

        # Compute average metrics
        avg_metrics = {metric: np.mean(values) for metric, values in total_metrics.items()}
        print("Evaluation loop complete.")
        return avg_metrics

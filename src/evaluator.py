import torch
import numpy as np
from src.metrics import compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from src.utils.helpers import map_classes_to_colors  # Assuming this function maps class IDs to RGB colors

class Evaluator:
    def __init__(self, model, device, class_to_color, metrics_config):
        """
        Initializes the evaluator.

        Args:
            model (torch.nn.Module): Trained model.
            device (torch.device): Device to run evaluation on.
            class_to_color (dict): Mapping of class IDs to RGB colors.
            metrics_config (dict): Configuration for metrics to evaluate.
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
            dict: Dictionary containing computed metrics (IoU, PixelAccuracy, DICE).
        """
        results = {}
        if self.metrics_config.get("iou", True):
            results["IoU"], results["MeanIoU"] = compute_iou(predictions, targets, self.num_classes)
        if self.metrics_config.get("pixel_accuracy", True):
            results["PixelAccuracy"] = compute_pixel_accuracy(predictions, targets)
        if self.metrics_config.get("dice", True):
            results["DICE"], results["MeanDICE"] = compute_dice_coefficient(predictions, targets, self.num_classes)
        return results

    def evaluate(self, data_loader):
        """
        Evaluate the model on the given data loader.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation dataset.

        Returns:
            dict: Aggregated metrics across all batches.
        """
        print("Starting evaluation loop...")
        self.model.eval()

        # Initialize total metrics including MeanIoU and MeanDICE
        total_metrics = {metric: [] for metric in self.metrics_config.keys()}
        total_metrics["MeanIoU"] = []
        total_metrics["MeanDICE"] = []

        # Storage for RGB conversions
        rgb_predictions_list = []
        rgb_targets_list = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, _, _) in enumerate(data_loader):
                print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs)
                predictions = torch.argmax(predictions, dim=1).cpu().numpy()
                targets = targets.cpu().numpy()

                # Compute metrics for the batch
                batch_metrics = self.evaluate_batch(predictions, targets)
                print(f"Batch {batch_idx + 1} Metrics: {batch_metrics}")

                # Aggregate metrics
                for metric, value in batch_metrics.items():
                    total_metrics[metric].append(value)

                # Convert class predictions and targets to RGB
                rgb_predictions = np.stack([
                    map_classes_to_colors(predictions[i], self.class_to_color) for i in range(predictions.shape[0])
                ])
                rgb_targets = np.stack([
                    map_classes_to_colors(targets[i], self.class_to_color) for i in range(targets.shape[0])
                ])

                rgb_predictions_list.append(rgb_predictions)
                rgb_targets_list.append(rgb_targets)

        # Compute average metrics, including MeanIoU and MeanDICE
        avg_metrics = {metric: np.mean(values) for metric, values in total_metrics.items()}
        print("Evaluation loop complete.")

        # Return metrics and RGB representations
        return avg_metrics, rgb_predictions_list, rgb_targets_list

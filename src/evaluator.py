import torch
import numpy as np
from src.metrics import compute_iou,compute_iou_for_target_classes_only, compute_pixel_accuracy, compute_dice_coefficient
from src.utils.mapping import classIndexToMask
import os
import matplotlib.pyplot as plt
# import segmentation_models_pytorch as smp

class Evaluator:
    def __init__(self, model, device,loss_fn, num_classes, metrics_config):
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
        self.loss_fn = loss_fn
        self.num_classes = num_classes
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
        predictions_tensor = torch.from_numpy(predictions)
        targets_tensor = torch.from_numpy(targets)
        results = {}
        if self.metrics_config.get("iou", False):
            # Get stats (TP, FP, FN, TN)
            # Move tensors to the same device as your model, e.g. 'cuda' if using GPUs
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predictions_tensor = predictions_tensor.to(device)
            targets_tensor = targets_tensor.to(device)
            # TP, FP, FN, TN = smp.metrics.get_stats(
            #     output=predictions_tensor,
            #     target=targets_tensor,
            #     mode="multiclass",
            #     ignore_index=-1,  # If needed, ignore certain index (e.g., background)
            #     num_classes=self.num_classes
            # )
            
            # Compute IoU using the stats
            # iou = smp.metrics.iou_score(TP, FP, FN, TN, reduction="micro")
            results["IoUoverall"] = 1
            results["IoU"], results["MeanIoU"] = compute_iou_for_target_classes_only(predictions, targets, self.num_classes)
        if self.metrics_config.get("pixel_accuracy", False):
            results["PixelAccuracy"] = compute_pixel_accuracy(predictions, targets)
        if self.metrics_config.get("dice", False):
            results["DICE"], results["MeanDICE"] = compute_dice_coefficient(predictions, targets, self.num_classes)
        return results

    def evaluate(self, data_loader, save_rgb=False, output_dir=None):
        """
        Evaluate the model on the given data loader and compute validation loss and accuracy.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation dataset.
            save_rgb (bool): Whether to save RGB visualizations of predictions and targets.
            output_dir (str): Directory to save RGB visualizations if save_rgb is True.

        Returns:
            dict: Aggregated metrics across all batches.
            float: Validation loss (average loss across all batches).
        """
        # print("Starting evaluation loop...")
        self.model.eval()

        # Initialize total metrics based on enabled configuration
        total_metrics = {}
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        batch_count = 0

        if self.metrics_config.get("iou", False):
            total_metrics["IoU"] = {cls: [] for cls in range(self.num_classes)}
            total_metrics["MeanIoU"] = []
            total_metrics["IoUoverall"] = []
        if self.metrics_config.get("pixel_accuracy", False):
            total_metrics["PixelAccuracy"] = []
        if self.metrics_config.get("dice", False):
            total_metrics["DICE"] = {cls: [] for cls in range(self.num_classes)}
            total_metrics["MeanDICE"] = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, _, _) in enumerate(data_loader):
                # print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                total_loss += self.loss_fn(outputs, targets).item()
                batch_count += 1

                # Predictions and ground truth
                predictions = torch.argmax(outputs, dim=1)  # Keep predictions on the same device
                total_correct += (predictions == targets).sum().item()
                total_pixels += targets.numel()

                # Convert to numpy for batch metrics
                predictions_np = predictions.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Compute batch metrics
                batch_metrics = self.evaluate_batch(predictions_np, targets_np)
                # print(f"Batch {batch_idx + 1} Metrics: {batch_metrics}")

                # Aggregate metrics
                for metric, values in batch_metrics.items():
                    if metric in ["IoU", "DICE"]:
                        for cls, value in enumerate(values):
                            if np.sum(targets_np == cls) > 0:  # Only update if class exists
                                total_metrics[metric][cls].append(value)
                    else:
                        total_metrics[metric].append(values)

                # Optionally save RGB visualizations
                if save_rgb and output_dir:
                    rgb_predictions = np.stack([
                        classIndexToMask(predictions_np[i]) for i in range(predictions_np.shape[0])
                    ])
                    rgb_targets = np.stack([
                        classIndexToMask(targets_np[i]) for i in range(targets_np.shape[0])
                    ])
                    for i, (rgb_pred, rgb_target) in enumerate(zip(rgb_predictions, rgb_targets)):
                        pred_path = os.path.join(output_dir, f"batch_{batch_idx + 1}sample{i}_pred.png")
                        target_path = os.path.join(output_dir, f"batch_{batch_idx + 1}sample{i}_target.png")
                        plt.imsave(pred_path, rgb_pred)
                        plt.imsave(target_path, rgb_target)

        # Compute average metrics
        avg_metrics = {}
        for metric, values in total_metrics.items():
            if metric in ["IoU", "DICE"]:
                avg_metrics[metric] = {
                    cls: (np.mean(cls_values) if cls_values else 0.0) for cls, cls_values in values.items()
                }
                avg_metrics[f"Mean{metric}"] = np.mean([
                    np.mean(cls_values) for cls_values in values.values() if cls_values
                ])
            else:
                avg_metrics[metric] = np.mean(values)

        # Compute validation loss and accuracy
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        val_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

        # print("Evaluation loop complete.")
        # print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        avg_metrics["val_loss"] = avg_loss
        avg_metrics["val_accuracy"] = val_accuracy

        return avg_metrics,avg_loss
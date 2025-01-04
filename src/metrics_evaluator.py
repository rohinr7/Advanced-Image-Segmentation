import numpy as np
from src.metrics import compute_iou, compute_pixel_accuracy, compute_dice_coefficient
import torch 
import time
import segmentation_models_pytorch as smp

class Evaluator:
    """
    A reusable class for evaluating segmentation models with various metrics.
    """
    def __init__(self, class_to_color, num_classes):
        """
        Args:
            class_to_color (dict): Mapping of class IDs to RGB colors.
            num_classes (int): Total number of classes.
        """
        self.class_to_color = class_to_color
        self.num_classes = num_classes

    def evaluate_batch(self, predictions, targets, metrics_config):
        """
        Evaluate metrics for a single batch.

        Args:
            predictions (np.ndarray): Predicted class IDs of shape (N, H, W).
            targets (np.ndarray): Ground truth class IDs of shape (N, H, W).
            metrics_config (dict): Configuration for which metrics to calculate.

        Returns:
            dict: Dictionary containing computed metrics (IoU, PixelAccuracy, DICE).
        """
        results = {}
        if metrics_config.get("iou", True):
            predictions_tensor = torch.tensor(predictions, dtype=torch.long)
            targets_tensor = torch.tensor(targets, dtype=torch.long)

            # Get stats (TP, FP, FN, TN)
            TP, FP, FN, TN = smp.metrics.get_stats(
                output=predictions_tensor,
                target=targets_tensor,
                mode="multiclass",
                ignore_index=-1,  # If needed, ignore certain index (e.g., background)
                num_classes=self.num_classes
            )
            
            # Compute IoU using the stats
            iou = smp.metrics.iou_score(TP, FP, FN, TN, reduction="micro")
            results["IoUoverall"] = iou
            results["IoU"], results["MeanIoU"] = compute_iou_for_target_classes_only(predictions, targets, self.num_classes)
        if metrics_config.get("pixel_accuracy", True):
            results["PixelAccuracy"] = compute_pixel_accuracy(predictions, targets)
        if metrics_config.get("dice", True):
            results["DICE"], results["MeanDICE"] = compute_dice_coefficient(predictions, targets, self.num_classes)

        return results


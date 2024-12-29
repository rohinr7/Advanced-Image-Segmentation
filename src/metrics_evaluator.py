import numpy as np
from src.metrics import compute_iou, compute_pixel_accuracy, compute_dice_coefficient
import time

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
            results["IoU"], results["MeanIoU"] = compute_iou(predictions, targets, self.num_classes)
        if metrics_config.get("pixel_accuracy", True):
            results["PixelAccuracy"] = compute_pixel_accuracy(predictions, targets)
        if metrics_config.get("dice", True):
            results["DICE"], results["MeanDICE"] = compute_dice_coefficient(predictions, targets, self.num_classes)
        return results


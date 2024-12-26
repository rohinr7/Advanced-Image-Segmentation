import numpy as np


def Precision():
    pass

def Recall():
    pass


def rgb_to_class_ids(rgb_image, class_to_color):
    """
    Convert an RGB image to class IDs based on the class-to-color mapping.

    Args:
        rgb_image (np.ndarray): RGB image of shape (H, W, 3).
        class_to_color (dict): Mapping from class IDs to RGB colors.

    Returns:
        np.ndarray: Class ID map of shape (H, W).
    """
    # Reverse mapping: RGB -> class ID
    color_to_class = {tuple(v): k for k, v in class_to_color.items()}
    height, width, _ = rgb_image.shape
    class_ids = np.zeros((height, width), dtype=np.int32)

    for rgb, class_id in color_to_class.items():
        mask = np.all(rgb_image == rgb, axis=-1)
        class_ids[mask] = class_id

    return class_ids


def compute_iou(predictions_rgb, targets_rgb, class_to_color, num_classes):
    """
    Compute IoU (Intersection over Union) for RGB predictions and RGB ground truth.

    Args:
        predictions_rgb (np.ndarray): RGB predictions of shape (N, H, W, 3).
        targets_rgb (np.ndarray): RGB ground truth of shape (N, H, W, 3).
        class_to_color (dict): Mapping from class IDs to RGB colors.
        num_classes (int): Number of classes.

    Returns:
        float: Mean IoU (mIoU).
    """
    iou_per_class = np.zeros(num_classes)

    for cls in range(num_classes):
        intersection = 0
        union = 0

        for i in range(predictions_rgb.shape[0]):
            # Convert RGB to class IDs
            pred_class_ids = rgb_to_class_ids(predictions_rgb[i], class_to_color)
            target_class_ids = rgb_to_class_ids(targets_rgb[i], class_to_color)

            # Calculate intersection and union
            intersection += np.sum((pred_class_ids == cls) & (target_class_ids == cls))
            union += np.sum((pred_class_ids == cls) | (target_class_ids == cls))

        # Compute IoU for the class
        iou_per_class[cls] = intersection / union if union > 0 else 0

    # Mean IoU
    mean_iou = np.mean(iou_per_class)
    return iou_per_class, mean_iou


def Pixel_accuracy():
    pass


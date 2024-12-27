import numpy as np


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


def compute_pixel_accuracy(predictions_rgb, targets_rgb, class_to_color):
    """
    Compute pixel accuracy for RGB predictions and RGB ground truth.

    Args:
        predictions_rgb (np.ndarray): RGB predictions of shape (N, H, W, 3).
        targets_rgb (np.ndarray): RGB ground truth of shape (N, H, W, 3).
        class_to_color (dict): Mapping from class IDs to RGB colors.

    Returns:
        float: Pixel accuracy score.
    """
    # Convert RGB to class IDs for predictions and targets
    color_to_class = {tuple(v): k for k, v in class_to_color.items()}
    total_correct = 0
    total_pixels = 0

    for i in range(predictions_rgb.shape[0]):
        pred_class_ids = rgb_to_class_ids(predictions_rgb[i], class_to_color)
        target_class_ids = rgb_to_class_ids(targets_rgb[i], class_to_color)

        # Count correctly predicted pixels
        total_correct += np.sum(pred_class_ids == target_class_ids)
        total_pixels += target_class_ids.size

    # Compute pixel accuracy
    pixel_accuracy = total_correct / total_pixels
    return pixel_accuracy


def compute_dice_coefficient(predictions_rgb, targets_rgb, class_to_color, num_classes):
    """
    Compute DICE coefficient for RGB predictions and RGB ground truth.

    Args:
        predictions_rgb (np.ndarray): RGB predictions of shape (N, H, W, 3).
        targets_rgb (np.ndarray): RGB ground truth of shape (N, H, W, 3).
        class_to_color (dict): Mapping from class IDs to RGB colors.
        num_classes (int): Number of classes.

    Returns:
        list: DICE score for each class.
        float: Mean DICE coefficient across all classes.
    """
    color_to_class = {tuple(v): k for k, v in class_to_color.items()}
    dice_per_class = []

    for cls in range(num_classes):
        intersection = 0
        pred_area = 0
        target_area = 0

        for i in range(predictions_rgb.shape[0]):
            pred_class_ids = rgb_to_class_ids(predictions_rgb[i], class_to_color)
            target_class_ids = rgb_to_class_ids(targets_rgb[i], class_to_color)

            intersection += np.sum((pred_class_ids == cls) & (target_class_ids == cls))
            pred_area += np.sum(pred_class_ids == cls)
            target_area += np.sum(target_class_ids == cls)

        dice = (2 * intersection) / (pred_area + target_area) if (pred_area + target_area) > 0 else 0
        dice_per_class.append(dice)

    mean_dice = np.mean(dice_per_class)
    return dice_per_class, mean_dice


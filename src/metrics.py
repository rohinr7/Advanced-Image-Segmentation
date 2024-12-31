import numpy as np

def rgb_to_class_ids(rgb_image, class_to_color):
    """
    Convert an RGB image to class IDs based on the class-to-color mapping.
    Optimized for vectorized operations.

    Args:
        rgb_image (np.ndarray): RGB image of shape (H, W, 3).
        class_to_color (dict): Mapping from class IDs to RGB colors.

    Returns:
        np.ndarray: Class ID map of shape (H, W).
    """
    color_to_class = {tuple(v): k for k, v in class_to_color.items()}
    reshaped_image = rgb_image.reshape(-1, 3)
    class_ids_flat = np.zeros(reshaped_image.shape[0], dtype=np.int32)

    for color, class_id in color_to_class.items():
        matches = np.all(reshaped_image == color, axis=1)
        class_ids_flat[matches] = class_id

    return class_ids_flat.reshape(rgb_image.shape[:2])


def compute_iou(predictions_class_ids, targets_class_ids, num_classes):
    """
    Compute IoU (Intersection over Union) for class IDs.

    Args:
        predictions_class_ids (np.ndarray): Predicted class IDs of shape (N, H, W).
        targets_class_ids (np.ndarray): Ground truth class IDs of shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: IoU for each class.
        float: Mean IoU (mIoU), ignoring classes with no samples in the target.
    """
    iou_per_class = np.zeros(num_classes, dtype=np.float32)
    valid_classes = np.zeros(num_classes, dtype=bool)  # Tracks classes with samples in the target.

    for cls in range(num_classes):
        pred_mask = (predictions_class_ids == cls)
        target_mask = (targets_class_ids == cls)

        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)

        if union > 0:
            iou_per_class[cls] = intersection / union
            valid_classes[cls] = True  # Mark as valid if there's at least one target sample.

    # Calculate mean IoU, ignoring invalid classes
    mean_iou = np.sum(iou_per_class[valid_classes]) / np.sum(valid_classes) if np.sum(valid_classes) > 0 else 0.0
    return iou_per_class, mean_iou



def compute_pixel_accuracy(predictions_class_ids, targets_class_ids):
    """
    Compute pixel accuracy for class IDs.

    Args:
        predictions_class_ids (np.ndarray): Predicted class IDs of shape (N, H, W).
        targets_class_ids (np.ndarray): Ground truth class IDs of shape (N, H, W).

    Returns:
        float: Pixel accuracy score.
    """
    total_correct = np.sum(predictions_class_ids == targets_class_ids)
    total_pixels = targets_class_ids.size
    return total_correct / total_pixels


def compute_dice_coefficient(predictions_class_ids, targets_class_ids, num_classes):
    """
    Compute DICE coefficient for class IDs.

    Args:
        predictions_class_ids (np.ndarray): Predicted class IDs of shape (N, H, W).
        targets_class_ids (np.ndarray): Ground truth class IDs of shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: DICE score for each class.
        float: Mean DICE coefficient across all valid classes.
    """
    dice_per_class = np.zeros(num_classes, dtype=np.float32)
    valid_classes = np.zeros(num_classes, dtype=bool)  # Tracks classes with samples in the target.

    for cls in range(num_classes):
        pred_mask = (predictions_class_ids == cls)
        target_mask = (targets_class_ids == cls)

        intersection = np.sum(pred_mask & target_mask)
        pred_area = np.sum(pred_mask)
        target_area = np.sum(target_mask)

        denominator = pred_area + target_area
        if denominator > 0:
            dice_per_class[cls] = (2 * intersection) / denominator
            valid_classes[cls] = True  # Mark as valid if there's at least one target sample.

    # Calculate mean DICE, ignoring invalid classes
    mean_dice = np.sum(dice_per_class[valid_classes]) / np.sum(valid_classes) if np.sum(valid_classes) > 0 else 0.0
    return dice_per_class, mean_dice
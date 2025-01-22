import numpy as np

# def rgb_to_class_ids(rgb_image, class_to_color):
#     """
#     Convert an RGB image to class IDs based on the class-to-color mapping.
#     Optimized for vectorized operations.

#     Args:
#         rgb_image (np.ndarray): RGB image of shape (H, W, 3).
#         class_to_color (dict): Mapping from class IDs to RGB colors.

#     Returns:
#         np.ndarray: Class ID map of shape (H, W).
#     """
#     color_to_class = {tuple(v): k for k, v in class_to_color.items()}
#     reshaped_image = rgb_image.reshape(-1, 3)
#     class_ids_flat = np.zeros(reshaped_image.shape[0], dtype=np.int32)

#     for color, class_id in color_to_class.items():
#         matches = np.all(reshaped_image == color, axis=1)
#         class_ids_flat[matches] = class_id

#     return class_ids_flat.reshape(rgb_image.shape[:2])

def compute_iou_for_target_classes_only(predictions, targets, num_classes):
    """
    Computes IoU for each class, but only for classes that actually appear in the target.
    Classes that never appear in the target are skipped from the mean IoU.

    Args:
        predictions (np.ndarray): Predicted class IDs of shape (N, H, W) or (H, W).
        targets     (np.ndarray): Ground truth class IDs of the same shape as `predictions`.
        num_classes (int)       : Number of classes.

    Returns:
        iou_per_class (np.ndarray): A float array of length `num_classes`, where
                                    classes that don't appear in the target are `NaN`.
        mean_iou      (float)     : Average IoU over only the classes that appear
                                    in the target.
    """
    # Ensure shape matches
    assert predictions.shape == targets.shape, (
        f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}."
    )

    # Initialize IoU array with NaNs
    # We'll fill in valid values only for classes that appear in the target.
    iou_per_class = np.full(num_classes, np.nan, dtype=np.float32)

    # Flatten if needed (for easier Boolean mask operations)
    pred_flat = predictions.reshape(-1)
    tgt_flat  = targets.reshape(-1)

    # Compute IoU class by class
    for cls_id in range(num_classes):
        # Check if this class appears in the target at all
        target_count = np.sum(tgt_flat == cls_id)
        if target_count == 0:
            # Class not in target; skip it by leaving iou_per_class[cls_id] as NaN
            continue

        # Otherwise, compute IoU
        pred_mask = (pred_flat == cls_id)
        tgt_mask  = (tgt_flat  == cls_id)

        intersection = np.sum(pred_mask & tgt_mask)
        union        = np.sum(pred_mask | tgt_mask)

        if union > 0:
            iou_per_class[cls_id] = intersection / union
        else:
            # A corner case theoretically shouldn't happen if target_count > 0,
            # but if it does, define IoU as 0 or handle accordingly.
            iou_per_class[cls_id] = 0.0

    # Now compute mean IoU, ignoring any NaNs (i.e., classes that never appeared in the target)
    valid_mask = ~np.isnan(iou_per_class)
    if np.any(valid_mask):
        mean_iou = np.mean(iou_per_class[valid_mask])
    else:
        # If no class is valid (e.g., target is empty), define mean IoU as 0 or NaN
        mean_iou = 0.0

    return iou_per_class, mean_iou
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
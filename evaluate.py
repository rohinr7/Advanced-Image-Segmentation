import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from src.dataset import ProjectDatasets
from torch.utils.data import DataLoader, random_split
from src.models.UNet import UNet
import random
import os
import numpy as np
import pandas as pd
from src.metrics import compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from src.utils.helpers import map_classes_to_colors, class_to_color, visualize_batch_with_colorbar

def evaluate(checkpoints_path, data_path, evaluate_iou=True, evaluate_pixel_accuracy=True, evaluate_dice=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = ProjectDatasets(root_path=data_path, transform=transform)

    model = UNet(in_channels=3, out_channels=30)
    model.to(device)

    # Pre-warm CUDA
    if torch.cuda.is_available():
        _ = torch.randn(1).to(device)

    checkpoint = torch.load(checkpoints_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Split dataset
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    iou_results = {f"Class_{cls}": [] for cls in range(len(class_to_color))}
    iou_results["Batch"] = []
    iou_results["Mean_IoU"] = []

    dice_results = {f"Class_{cls}": [] for cls in range(len(class_to_color))}
    dice_results["Batch"] = []
    dice_results["Mean_DICE"] = []

    total_iou_per_class = np.zeros(len(class_to_color))
    total_dice_per_class = np.zeros(len(class_to_color))
    total_batches = 0

    total_pixel_accuracy = 0
    num_batches = 0

    for batch_idx, (inputs, targets, _) in enumerate(val_loader):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            predictions = torch.argmax(predictions, dim=1)  # Shape: [N, H, W]

            # Convert predictions and targets to RGB format
            rgb_predictions = np.stack([
                map_classes_to_colors(predictions[i].cpu().numpy(), class_to_color) for i in range(predictions.shape[0])
            ])  # Shape: [N, H, W, 3]

            rgb_targets = np.stack([
                map_classes_to_colors(targets[i].cpu().numpy(), class_to_color) for i in range(targets.shape[0])
            ])  # Shape: [N, H, W, 3]

            if evaluate_iou:
                # Compute IoU for the current batch
                batch_iou_scores, batch_mean_iou = compute_iou(rgb_predictions, rgb_targets, class_to_color, len(class_to_color))
                print(f"The Mean IoU score of the batch {batch_idx} is {batch_mean_iou}")

                # Store IoU for each class
                for cls, iou in enumerate(batch_iou_scores):
                    iou_results[f"Class_{cls}"].append(iou)
                    total_iou_per_class[cls] += iou

                # Store batch information
                iou_results["Batch"].append(batch_idx)
                iou_results["Mean_IoU"].append(batch_mean_iou)
                total_batches += 1

            if evaluate_pixel_accuracy:
                # Compute pixel accuracy for the current batch
                batch_pixel_accuracy = compute_pixel_accuracy(rgb_predictions, rgb_targets, class_to_color)
                total_pixel_accuracy += batch_pixel_accuracy
                num_batches += 1

                print(f"Batch {batch_idx} Pixel Accuracy: {batch_pixel_accuracy:.4f}")

            if evaluate_dice:
                # Compute DICE coefficient for the current batch
                batch_dice_scores, batch_mean_dice = compute_dice_coefficient(rgb_predictions, rgb_targets, class_to_color, len(class_to_color))
                print(f"The Mean DICE score of the batch {batch_idx} is {batch_mean_dice}")

                # Store DICE for each class
                for cls, dice in enumerate(batch_dice_scores):
                    dice_results[f"Class_{cls}"].append(dice)
                    total_dice_per_class[cls] += dice

                # Store batch information
                dice_results["Batch"].append(batch_idx)
                dice_results["Mean_DICE"].append(batch_mean_dice)
                if evaluate_iou == False:
                    total_batches += 1

    if evaluate_iou:
        # Compute overall IoU per class and overall mean IoU
        overall_iou_per_class = total_iou_per_class / total_batches
        overall_mean_iou = sum(overall_iou_per_class) / len(overall_iou_per_class)

        # Append overall IoU results to the DataFrame
        iou_results["Batch"].append("Overall")
        iou_results["Mean_IoU"].append(overall_mean_iou)
        for cls, overall_iou in enumerate(overall_iou_per_class):
            iou_results[f"Class_{cls}"].append(overall_iou)

        # Convert to DataFrame and save to CSV
        iou_df = pd.DataFrame(iou_results)
        iou_csv_path = "experiments/evaluation/validation_iou_results.csv"
        iou_df.to_csv(iou_csv_path, index=False)

    if evaluate_pixel_accuracy:
        # Compute overall pixel accuracy for the validation set
        overall_pixel_accuracy = total_pixel_accuracy / num_batches
        print(f"Overall Pixel Accuracy for Validation Set: {overall_pixel_accuracy:.4f}")

    if evaluate_dice:
        # Compute overall DICE coefficient per class and overall mean DICE coefficient
        overall_dice_per_class = total_dice_per_class / total_batches
        overall_mean_dice = sum(overall_dice_per_class) / len(overall_dice_per_class)

        # Append overall DICE results to the DataFrame
        dice_results["Batch"].append("Overall")
        dice_results["Mean_DICE"].append(overall_mean_dice)
        for cls, overall_dice in enumerate(overall_dice_per_class):
            dice_results[f"Class_{cls}"].append(overall_dice)

        # Convert to DataFrame and save to CSV
        dice_df = pd.DataFrame(dice_results)
        dice_csv_path = "experiments/evaluation/validation_dice_results.csv"
        dice_df.to_csv(dice_csv_path, index=False)


if __name__ == "__main__":
    checkpoints_path = "/net/travail/rramesh/AdvanceimageProcessing/Semantic-Segmentation-for-Autonomous-Driving/experiments/checkpoint_epoch_10.pth"
    data_path =  "/net/ens/am4ip/datasets/project-dataset"
 
    evaluate(checkpoints_path, data_path, evaluate_iou=False, evaluate_pixel_accuracy=False, evaluate_dice=True)

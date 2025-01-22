import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, weight=None):
        """
        Initializes the DiceLoss module.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            weight (torch.Tensor, optional): Tensor of shape [num_classes] assigning weight to each class.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight  # Expecting shape [num_classes]

    def forward(self, logits, targets):
        """
        Computes the Dice loss.

        Args:
            logits (torch.Tensor): Predicted probabilities with shape [batch_size, num_classes, H, W].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, H, W].

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        # Assuming logits are already softmaxed
        probs = logits  # [batch_size, num_classes, H, W]

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])  # [batch_size, H, W, num_classes]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()     # [batch_size, num_classes, H, W]

        # Calculate intersection and union per class
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))      # [batch_size, num_classes]
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))  # [batch_size, num_classes]

        # Compute Dice score per class
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)  # [batch_size, num_classes]

        # Apply weights if provided
        if self.weight is not None:
            # Ensure weight is on the same device and type as dice_score
            weight = self.weight.to(dice_score.device).type_as(dice_score)
            dice_score = dice_score * weight  # [batch_size, num_classes]

        # Compute Dice loss
        dice_loss = 1 - dice_score  # [batch_size, num_classes]

        # Average over classes and batch
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        self.dice_loss = DiceLoss()  # Removed mode='multiclass'
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.weight * dice + (1 - self.weight) * ce


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction="mean"):
        """
        Initializes the FocalLoss module.

        Args:
            alpha (float or torch.Tensor, optional): Weighting factor for the focal loss. 
                If a tensor, it should have shape [num_classes].
            gamma (float): Focusing parameter that reduces the relative loss for well-classified examples.
            weight (torch.Tensor, optional): Tensor of shape [num_classes] assigning weight to each class.
            reduction (str): Specifies the reduction to apply to the output: "none" | "mean" | "sum".
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # Expecting shape [num_classes]
        self.reduction = reduction

        # Initialize CrossEntropyLoss with provided weights
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", weight=self.weight)

    def forward(self, logits, targets):
        """
        Computes the Focal loss.

        Args:
            logits (torch.Tensor): Predicted logits with shape [batch_size, num_classes, H, W].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, H, W].

        Returns:
            torch.Tensor: Scalar Focal loss.
        """
        # Compute CrossEntropyLoss without reduction
        ce_loss = self.ce_loss(logits, targets)  # [batch_size, H, W]

        # Compute softmax probabilities
        probs = logits

        # Gather the probabilities corresponding to the targets
        targets_unsqueezed = targets.unsqueeze(1)  # [batch_size, 1, H, W]
        p_t = probs.gather(1, targets_unsqueezed).squeeze(1)  # [batch_size, H, W]

        # Compute the focal loss component
        focal_component = (1 - p_t) ** self.gamma  # [batch_size, H, W]

        # Compute the focal loss
        focal_loss = self.alpha * focal_component * ce_loss  # [batch_size, H, W]

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
    
class MeanIoULoss(nn.Module):
    def __init__(self, num_classes=34, epsilon=1e-6, ignore_index=[1,2,3,4,5,6,9,10,14,15,16,17]):
        """
        Initializes the MeanIoULoss module.

        Args:
            num_classes (int): Number of classes in the dataset.
            epsilon (float): Small constant to avoid division by zero.
            ignore_index (list[int] or int, optional): Class indices to ignore in loss computation.
        """
        super(MeanIoULoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        if isinstance(ignore_index, int):  # Allow single integer as input
            ignore_index = [ignore_index]
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Computes the Mean IoU Loss.

        Args:
            logits (torch.Tensor): Predicted logits with shape [batch_size, num_classes, H, W].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, H, W].

        Returns:
            torch.Tensor: Mean IoU loss.
        """
        # Apply softmax to logits to get probabilities
        probs = torch.softmax(logits, dim=1)  # [batch_size, num_classes, H, W]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # [batch_size, H, W, num_classes]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()       # [batch_size, num_classes, H, W]

        # Initialize intersection and union for each class
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))    # [num_classes]
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(targets_one_hot, dim=(0, 2, 3))  # [num_classes]

        # Compute IoU for each class
        iou = intersection / (union - intersection + self.epsilon)  # [num_classes]

        # Handle ignore_index if specified
        if self.ignore_index is not None:
            mask = torch.ones_like(iou, dtype=torch.bool)
            for idx in self.ignore_index:
                mask[idx] = False
            iou = iou[mask]

        # Compute mean IoU
        mean_iou = torch.mean(iou)

        # Loss is 1 - mean IoU
        return 1 - mean_iou


class DenoiceLosses:
    def __init__(self, vgg_model, lambda_adv=1.0, lambda_perc=1.0):
        self.vgg = vgg_model
        self.lambda_adv = lambda_adv
        self.lambda_perc = lambda_perc
        self.mse_loss = nn.MSELoss()

    def perceptual_loss(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return sum(F.l1_loss(p, t) for p, t in zip(pred_features, target_features))

    def total_loss(self, pred, target, gen_output=None, real_output=None):
        adv_loss = 0.0
        if gen_output is not None:
            adv_loss = self.mse_loss(gen_output, torch.ones_like(gen_output))

        perc_loss = self.perceptual_loss(pred, target)
        pix_loss = F.l1_loss(pred, target)

        return pix_loss + self.lambda_adv * adv_loss + self.lambda_perc * perc_loss, adv_loss, perc_loss, pix_loss

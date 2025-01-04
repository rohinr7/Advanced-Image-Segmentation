import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Since softmax is already applied inside the model, no need to apply softmax here
        probs = logits  # Directly use the model output (already softmaxed)
        
        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))  # Sum over height and width
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        
        # Dice score formula
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Return the average Dice loss
        return 1 - dice.mean()


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
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        tp = torch.sum(probs * targets_one_hot, dim=(2, 3))
        fp = torch.sum(probs * (1 - targets_one_hot), dim=(2, 3))
        fn = torch.sum((1 - probs) * targets_one_hot, dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()
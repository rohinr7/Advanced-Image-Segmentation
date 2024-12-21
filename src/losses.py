
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        intersection = (predictions * targets).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return bce_loss + dice_loss

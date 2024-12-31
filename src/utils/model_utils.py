# src/utils/model_utils.py
import torch
from src.models.unet import UNet
from src.models.deeplabv3plus import DeepLabV3Plus

def get_model(config):
    if config["model"]["name"] == "UNet":
        return UNet(
            in_channels=config["hyperparameters"]["input_channels"],
            out_channels=config["hyperparameters"]["output_channels"],
        )
    elif config["model"]["name"] == "DeepLabV3Plus":
        return DeepLabV3Plus(
            num_classes=config["hyperparameters"]["output_channels"],
            in_channels=config["hyperparameters"]["input_channels"],
            backbone=config["model"].get("backbone", "resnet"),
            pretrained=config["model"].get("pretrained", True),
        )
    else:
        raise ValueError(f"Unsupported model: {config['model']['name']}")

def get_loss_function(config):
    """
    Returns the appropriate loss function based on the configuration.
    """
    loss_name = config["loss"]["name"]
    params = config["loss"].get("params", {})

    if loss_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(**params)
    elif loss_name == "DiceLoss":
        from src.losses import DiceLoss
        return DiceLoss(**params)
    elif loss_name == "FocalLoss":
        from src.losses import FocalLoss
        return FocalLoss(**params)
    elif loss_name == "CombinedLoss":
        from src.losses import CombinedLoss
        return CombinedLoss(**params)
    elif loss_name == "TverskyLoss":
        from src.losses import TverskyLoss
        return TverskyLoss(**params)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

def get_optimizer(config, model):
    optimizer_name = config["optimizer"]["name"]
    params = config["optimizer"]["params"]
    
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), **params)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), **params)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **params)  # Added support for AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(config, optimizer):
    scheduler_name = config["scheduler"]["name"]
    params = config["scheduler"]["params"]
    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

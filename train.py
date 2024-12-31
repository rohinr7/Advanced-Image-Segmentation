import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
import random
import numpy as np

from src.dataset import ProjectDatasets
from src.trainer import Trainer
from src.models.unet import UNet
from src.models.deeplabv3plus import DeepLabV3Plus
from src.utils.helpers import save_checkpoint, load_checkpoint, custom_collate_fn, save_config,load_config,log_metrics,log_metrics_test,log_to_file
from src.metrics import compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop

from src.utils.model_utils import get_model, get_loss_function, get_optimizer, get_scheduler
from src.utils.transform_utils import get_transforms
from src.utils.helpers import load_config
from src.evaluator import Evaluator


# # Helper functions for dynamic model, loss, optimizer, and scheduler setup
# def get_model(config):
#     if config["model"]["name"] == "UNet":
#         return UNet(
#             in_channels=config["hyperparameters"]["input_channels"],
#             out_channels=config["hyperparameters"]["output_channels"],
#         )
#     elif config["model"]["name"] == "DeepLabV3Plus":
#         return DeepLabV3Plus(
#             in_channels=config["hyperparameters"]["input_channels"],
#             out_channels=config["hyperparameters"]["output_channels"],
#         )
#     else:
#         raise ValueError(f"Unsupported model: {config['model']['name']}")

# def get_loss_function(config):
#     loss_name = config["loss"]["name"]
#     params = config["loss"].get("params", {})
#     if loss_name == "CrossEntropyLoss":
#         return torch.nn.CrossEntropyLoss(**params)
#     elif loss_name == "FocalLoss":
#         from src.losses import FocalLoss
#         return FocalLoss(**params)
#     else:
#         raise ValueError(f"Unsupported loss function: {loss_name}")

# def get_optimizer(config, model):
#     optimizer_name = config["optimizer"]["name"]
#     params = config["optimizer"]["params"]
#     if optimizer_name == "Adam":
#         print(model.parameters())
#         return torch.optim.Adam(model.parameters(), **params)
#     elif optimizer_name == "SGD":
#         return torch.optim.SGD(model.parameters(), **params)
#     else:
#         raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# def get_scheduler(config, optimizer):
#     scheduler_name = config["scheduler"]["name"]
#     params = config["scheduler"]["params"]
#     if scheduler_name == "StepLR":
#         return torch.optim.lr_scheduler.StepLR(optimizer, **params)
#     elif scheduler_name == "CosineAnnealingLR":
#         return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
#     else:
#         raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
# def get_transforms(config):
#     transforms = [ToTensor()]  # Convert to tensor
#     if config["augmentation"]["use_augmentation"]:
#         aug_params = config["augmentation"]["params"]
#         if aug_params.get("horizontal_flip", False):
#             transforms.append(RandomHorizontalFlip())  # Horizontal flip
#         if aug_params.get("rotation_range", 0) > 0:
#             transforms.append(RandomRotation(aug_params["rotation_range"]))  # Random rotation
#         if aug_params.get("brightness_adjust", 0) > 0:
#             transforms.append(ColorJitter(brightness=aug_params["brightness_adjust"]))  # Brightness adjustment
#     # Add normalization
#     transforms.append(Normalize(mean=[0.5], std=[0.5]))  
#     return Compose(transforms)

def main(args):
 
    # Load configuration
    config = load_config(args.config)

    # Access seed value
    seed = config["hyperparameters"]["seed"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
#    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform = get_transforms(config["augmentation"])

    # Load the full dataset
    full_dataset = ProjectDatasets(root_path=config["paths"]["data"], transform=transform)

    # Calculate split sizes based on config
    split_ratios = config["dataset_split"]  # e.g., {"train": 0.8, "val": 0.1, "test": 0.1}
    train_size = int(split_ratios["train"] * len(full_dataset))
    val_size = int(split_ratios["val"] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["hyperparameters"]["batch_size"],
        shuffle=config["data"].get("shuffle", True),
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),

    )
    val_loader = DataLoader(val_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    model = get_model(config)
    model = model.to(device)

    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer)

    # Initialize Trainer
    # trainer = Trainer(
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     device=device,
    #     experiment_name=args.experiment_name,
    #     early_stopping_config=config.get("early_stopping", None), 
    #     logging_config=config.get("logging", {}),
    #     class_to_color=config["class_to_color"]
    # )

    trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    experiment_name=args.experiment_name,
    logging_config=config["logging"],
    early_stopping_config=config["early_stopping"],
    metrics_config=config["metrics"],
    class_to_color=config["class_to_color"],
    class_names=config["class_names"]  # List of class names

)
    # Log experiment details
    log_to_file(f"Model: {config['model']['name']}", trainer.experiment_dir)
    log_to_file(f"Loss Function: {config['loss']['name']}", trainer.experiment_dir)
    log_to_file(f"Optimizer: {config['optimizer']['name']}", trainer.experiment_dir)
    log_to_file(f"Learning Rate: {config['optimizer']['params']['lr']}", trainer.experiment_dir)
    log_to_file(f"Scheduler: {config['scheduler']['name']}", trainer.experiment_dir)
    log_to_file(f"Scheduler Params: {config['scheduler']['params']}", trainer.experiment_dir)
    log_to_file(f"Batch Size: {config['hyperparameters']['batch_size']}", trainer.experiment_dir)
    log_to_file(f"Epochs: {config['hyperparameters']['epochs']}", trainer.experiment_dir)
    log_to_file(f"Seed: {config['hyperparameters']['seed']}", trainer.experiment_dir)

    # Log dataset details
    log_to_file("Dataset Details:", trainer.experiment_dir)
    log_to_file(f"Dataset Path: {config['paths']['data']}", trainer.experiment_dir)
    log_to_file(f"Train Split: {config['dataset_split']['train']}", trainer.experiment_dir)
    log_to_file(f"Validation Split: {config['dataset_split']['val']}", trainer.experiment_dir)
    log_to_file(f"Test Split: {config['dataset_split']['test']}", trainer.experiment_dir)

    # Log augmentation details
    log_to_file("Augmentation Details:", trainer.experiment_dir)
    log_to_file(f"Use Augmentation: {config['augmentation']['use_augmentation']}", trainer.experiment_dir)
    if config['augmentation']['use_augmentation']:
        for aug, value in config['augmentation']['params'].items():
            log_to_file(f"  {aug}: {value}", trainer.experiment_dir)

    # Save configuration
    save_config(config, trainer.experiment_dir)

    # Train and Validate
    trainer.fit(train_loader, val_loader, epochs=config["hyperparameters"]["epochs"], start_epoch=start_epoch)

    print("Runing a test set.")

    test_loader = DataLoader(test_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model=model, device=device,loss_fn=get_loss_function(config), class_to_color=config["class_to_color"], metrics_config=config["metrics"])
    metrics,_ = evaluator.evaluate(test_loader)
    log_metrics_test( trainer.experiment_dir,metrics, class_names=config["class_names"], indent=2, decimals=6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for segmentation")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to save/load checkpoint")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    main(args)
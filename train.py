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

from src.utils.mapping import CLASS_TO_COLOR

from datetime import datetime, timedelta

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
    #     class_to_color=CLASS_TO_COLOR
    # )
    num_classes=config["hyperparameters"]["output_channels"]

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
    num_classes=num_classes,
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

    # Start Training Timer
    train_start_time = datetime.now()
    print(f"Training started at {train_start_time}")

    # Train and Validate
    trainer.fit(train_loader, val_loader, epochs=config["hyperparameters"]["epochs"], start_epoch=start_epoch)

    # End Training Timer
    train_end_time = datetime.now()
    print(f"Training ended at {train_end_time}")
    train_duration = train_end_time - train_start_time
    print(f"Training duration: {str(train_duration)}")
    log_to_file(f"Training started at: {train_start_time}", trainer.experiment_dir)
    log_to_file(f"Training ended at: {train_end_time}", trainer.experiment_dir)
    log_to_file(f"Training duration: {str(train_duration)}", trainer.experiment_dir)


    print("Runing a test set.")

    test_loader = DataLoader(test_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=False, num_workers=4)
    # Start Testing Timer
    test_start_time = datetime.now()
    print(f"Testing started at {test_start_time}")
    evaluator = Evaluator(model=model, device=device,loss_fn=get_loss_function(config), num_classes=num_classes, metrics_config=config["metrics"])
    metrics,_ = evaluator.evaluate(test_loader)
    # End Testing Timer
    test_end_time = datetime.now()
    log_metrics_test( trainer.experiment_dir,metrics, class_names=config["class_names"], indent=2, decimals=6)
    print(f"Testing ended at {test_end_time}")
    test_duration = test_end_time - test_start_time
    print(f"Testing duration: {str(test_duration)}")
    log_to_file(f"Testing started at: {test_start_time}", trainer.experiment_dir)
    log_to_file(f"Testing ended at: {test_end_time}", trainer.experiment_dir)
    log_to_file(f"Testing duration: {str(test_duration)}", trainer.experiment_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for segmentation")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to save/load checkpoint")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    main(args)
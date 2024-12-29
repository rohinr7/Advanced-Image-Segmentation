import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.optim as optim
from src.dataset import ProjectDatasets
from src.trainer import Trainer
from src.models.UNet import UNet
from src.utils.helpers import save_checkpoint, load_checkpoint, custom_collate_fn, save_config
import random
import numpy as np


def main(args):
    # Set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

    # Load the full dataset
    full_dataset = ProjectDatasets(root_path=args.data_path, transform=transform)

    # Calculate split sizes
    train_size = int(0.8 * len(full_dataset))  # 80% for training
    val_size = len(full_dataset) - train_size  # Remaining 20% for validation

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    model = UNet(in_channels=3, out_channels=30)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        experiment_name=args.experiment_name,
    )

    # Save configuration
    save_config(vars(args),trainer.experiment_dir)

    # Train and Validate
    trainer.fit(train_loader, val_loader, epochs=args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model for segmentation")
    parser.add_argument("--data_path", type=str, default="/net/ens/am4ip/datasets/project-dataset", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--checkpoint_path", type=str, default="unet_checkpoint.pth", help="Path to save/load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
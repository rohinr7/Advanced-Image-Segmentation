# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.Denoisingarchitecture import DeRainNet
from src.trainer import DenoisingTrainer
from src.losses import DenoiceLosses
from src.dataset import RainyDataset  # Assumes your dataset loader is in src.dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import vgg19


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transform(image_size=256):
    """
    Returns the transformations for the dataset.

    Args:
        image_size (int): The size to which images should be resized.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),  # Resize images to (256, 256)
        T.ToTensor(),  # Convert to tensor
        # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])


def main():
    # Configuration
    config = {
        'dataset_path': '/net/travail/rramesh/RAIN_DATASET',
        'batch_size': 16,
        'lr': 0.0002,
        'epochs': 100,
        'val_split': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './experiments/denoising'
    }

    # Transformations for the dataset
    transform = get_transform(image_size=256)

    # Load dataset
    full_dataset = RainyDataset(root_path=config['dataset_path'], transform=transform)
    dataset_size = len(full_dataset)
    val_size = int(config['val_split'] * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = DeRainNet().to(config['device'])

    # Initialize VGG model for perceptual loss
    vgg_model = vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features[:16].eval()
    for param in vgg_model.parameters():
        param.requires_grad = False
    vgg_model = vgg_model.to(config['device'])

    # Loss function
    criterion = DenoiceLosses(vgg_model=vgg_model, lambda_adv=1.0, lambda_perc=1.0)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Initialize Trainer
    trainer = DenoisingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(config['device']),
        save_dir=config['save_dir']
    )

    # Start training
    trainer.fit(config['epochs'])

if __name__ == "__main__":
    main()
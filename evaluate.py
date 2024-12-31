import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import ProjectDatasets
from src.utils.model_utils import get_model,get_loss_function
from src.utils.transform_utils import get_transforms
from src.utils.helpers import load_config
from src.evaluator import Evaluator

def evaluate(config, model_path):
    """
    Evaluate the model on a validation or test dataset using specified metrics.

    Args:
        config (dict): Configuration dictionary loaded from YAML.
        model_path (str): Path to the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and transforms
    print("Loading dataset and applying transforms...")
    transform = get_transforms(config["augmentation"])
    dataset = ProjectDatasets(root_path=config["paths"]["data"], transform=transform)
    print(f"Total dataset size: {len(dataset)}")

    # Dataset split
    split_config = config["dataset_split"]
    train_size = int(split_config["train"] * len(dataset))
    val_size = int(split_config["val"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")

    generator = torch.Generator().manual_seed(config["hyperparameters"]["seed"])
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False)
    print(f"Test loader created with batch size: {config['evaluation']['batch_size']}")

    # Model initialization
    model = get_model(config)
    print("Loaded model state from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)
    print("Model successfully loaded and moved to device.")

    # Evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(model=model, device=device,loss_fn=get_loss_function(config), class_to_color=config["class_to_color"], metrics_config=config["metrics"])
    metrics,_ = evaluator.evaluate(test_loader)
    print(f"Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.model_path)

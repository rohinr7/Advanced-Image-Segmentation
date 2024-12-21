import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from src.dataset import ProjectDatasets
from src.models.UNet import UNet
from src.utils.metrics import compute_iou
from src.utils.visualization import visualize_predictions

def evaluate(model_path, data_path):
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    test_dataset = ProjectDatasets(root_path=data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    iou_scores = []

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = outputs > 0.5
            iou_scores.append(compute_iou(predictions, targets))

    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"Average IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate UNet model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_path)

import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.helpers import save_config, log_metrics, save_checkpoint, save_predictions

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device=None, experiment_name=None):
        self.model = model.to(device or "cpu")
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or "cpu"
        self.writer = SummaryWriter()  # TensorBoard writer
        self.writer.add_text("Device Info", f"Running on: {device}")

        # Create experiment folder
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_dir = os.path.join("experiments", experiment_name or f"experiment_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "results"), exist_ok=True)

    def fit(self, train_loader, val_loader, epochs, start_epoch=0):
        """Train the model."""
        self.model.train()
        for epoch in range(start_epoch + 1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            epoch_loss = 0.0

            for batch_idx, (inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                # unique_targets = targets.unique()
                # print(f"Unique target values: {unique_targets}")
                # if (unique_targets < 0).any() or (unique_targets >= 30).any():
                #     raise ValueError(f"Invalid target values: {unique_targets}")

                # Compute loss
                loss = self.loss_fn(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"[Batch {batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)


            # Validate
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch} - Avg Loss: {avg_epoch_loss:.4f} , Validation Loss: {val_loss:.4f}")


            # Save metrics and checkpoint
            log_metrics(epoch, avg_epoch_loss, val_loss,self.experiment_dir)
            save_checkpoint(self.model,self.optimizer,epoch,self.experiment_dir)

            # Save example predictions
            self.model.eval()
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    predictions = self.model(inputs)
                    save_predictions(inputs, targets, predictions, epoch,self.experiment_dir)
                    break  # Save only one batch
            self.model.train()
            
    def validate(self, val_loader):
        """Evaluate the model on the validation dataset."""
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()

        # Return average validation loss
        return val_loss / len(val_loader)
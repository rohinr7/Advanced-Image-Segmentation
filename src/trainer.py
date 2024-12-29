import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.helpers import save_config, log_metrics, save_checkpoint, save_predictions

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

import logging

def setup_logging(log_level="INFO"):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(self, model, loss_fn, optimizer,scheduler=None, device=None, experiment_name=None, logging_config=None, early_stopping_config=None):
        
        setup_logging(log_level=logging_config.get("log_level", "INFO"))
        self.tensorboard_enabled = logging_config.get("tensorboard", True)

        self.model = model.to(device or "cpu")
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler  # Add scheduler
        self.device = device or "cpu"
        self.writer = SummaryWriter()  # TensorBoard writer
        self.writer.add_text("Device Info", f"Running on: {device}")

        # ealy stop

        # Ensure early_stopping_config is a dictionary
        early_stopping_config = early_stopping_config or {}
        
        # Early stopping parameters
        self.early_stopping_enabled = early_stopping_config.get("enabled", False)
        self.patience = early_stopping_config.get("patience", 5)
        self.metric = early_stopping_config.get("metric", "val_loss")
        self.best_metric = float("inf") if self.metric == "val_loss" else float("-inf")
        self.epochs_no_improve = 0
        self.early_stop = False


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

            for batch_idx, (inputs, targets, _ , _ ) in enumerate(train_loader):
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
            
            # save_checkpoint(self.model,self.optimizer,epoch,self.experiment_dir)
            if val_metric < self.best_metric:
                save_checkpoint(self.model, self.optimizer, epoch, self.experiment_dir, is_best=True)


            # Save example predictions
            self.model.eval()
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    predictions = self.model(inputs)
                    save_predictions(inputs, targets, predictions, epoch,self.experiment_dir)
                    break  # Save only one batch
            self.model.train()
            
            val_metric = self.validate(val_loader)
            # Early stopping logic
            if self.metric == "val_loss" and val_metric < self.best_metric:
                self.best_metric = val_metric
                self.epochs_no_improve = 0
            elif self.metric != "val_loss" and val_metric > self.best_metric:
                self.best_metric = val_metric
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True
                break
            
    def validate(self, val_loader):
        """Evaluate the model on the validation dataset."""
        if self.early_stop:
            print("Early stopping triggered. Exiting training.")
            return
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
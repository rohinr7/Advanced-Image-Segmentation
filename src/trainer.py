import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.helpers import save_config, log_metrics, save_checkpoint, save_predictions,log_metrics,log_to_file
from src.evaluator import Evaluator
import numpy as np
import os
from datetime import datetime
import logging
import datetime
from tqdm import tqdm 
import matplotlib.pyplot as plt

def setup_logging(log_level="INFO"):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')



class Trainer:
    def __init__(self, model, loss_fn, optimizer,hyperparameters=None, scheduler=None, device=None, experiment_name=None, logging_config=None, early_stopping_config=None, metrics_config=None, num_classes=None,class_names=None):
        setup_logging(log_level=logging_config.get("log_level", "INFO"))
        self.tensorboard_enabled = logging_config.get("tensorboard", True)

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get the current timestamp
        self.experiment_dir = os.path.join("experiments", f"{experiment_name}_{timestamp}" if experiment_name else f"experiment_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.experiment_dir) if self.tensorboard_enabled else None
        self.hyperparameters = hyperparameters
        self.metrics_config = metrics_config
        self.num_classes = num_classes
        if class_names is None:
            raise ValueError("class_names must be provided in the Trainer constructor.")
        self.class_names = class_names
       #self.class_names = class_names or [f"Class_{i}" for i in range(len(class_to_color))]


        # Early stopping parameters
        early_stopping_config = early_stopping_config or {}
        self.early_stopping_enabled = early_stopping_config.get("enabled", False)
        self.patience = early_stopping_config.get("patience", 5)
        self.metric = early_stopping_config.get("metric", "val_loss")
        self.best_metric = float("inf") if self.metric == "val_loss" else float("-inf")
        self.epochs_no_improve = 0
        self.early_stop = False

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "results"), exist_ok=True)

    def log_model_info(self):
        print(f"Optimizer: {self.optimizer}")
        """Log model information (e.g., model name, loss function, optimizer) to TensorBoard."""
        model_info = f"Model: {self.model.__class__.__name__}\n"
        model_info += f"Loss Function: {self.loss_fn.__class__.__name__}\n"
        # model_info += f"Optimizer: {self.optimizer.__class__.__name__}\n"
        
        # If a scheduler exists, log its name, otherwise, mention "None"
        if self.scheduler:
            model_info += f"Scheduler: {self.scheduler.__class__.__name__}\n"
        else:
            model_info += "Scheduler: None\n"

        # Log model info as text in TensorBoard
        self.writer.add_text("Model Information", model_info)

        # Add optimizer name to the hyperparameters dictionary
        hyperparameters_with_optimizer = {**self.hyperparameters, "optimizer_name": self.optimizer.__class__.__name__}

        # Log hyperparameters including optimizer name using add_hparams
        self.writer.add_hparams(
            hyperparameters_with_optimizer, 
            {"hparam/metric": 0.0}  # You can replace this with the actual metric if needed
        )
    def compute_gradient_norm(self):
        """
        Compute the gradient norm for the model parameters.
        
        Returns:
            float: The L2 norm of gradients across all model parameters.
        """
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5  # Square root of the sum of squares
        return total_norm
    def log_learning_rate(self, epoch):
        """Log the current learning rate to TensorBoard."""
        if self.scheduler:
            current_lr = self.scheduler.get_last_lr()[0]  # Assuming one LR for simplicity
        else:
            current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Learning Rate", current_lr, epoch)

    def log_confidence_histogram(self, outputs, epoch):
        """Log the softmax confidence histogram."""
        # Validate output shape
        print(f"Shape of outputs before softmax: {outputs.shape}")
        
        # Resize if too large to avoid computational overhead
        if outputs.shape[2] > 128 or outputs.shape[3] > 128:  # Check height/width
            outputs = torch.nn.functional.interpolate(outputs, size=(128, 128), mode="bilinear", align_corners=False)
            print(f"Resized outputs to: {outputs.shape}")
        
        # Detach outputs, apply softmax, and compute maximum confidences
        probabilities = torch.softmax(outputs.detach(), dim=1).cpu().numpy()  # Detach from computation graph
        max_confidences = np.max(probabilities, axis=1).flatten()  # Shape: [N * H * W]
        
        # Log histogram
        self.writer.add_histogram("Confidence/Max_Confidence", max_confidences, epoch)



    def log_class_distribution(self, predictions, epoch):
        """Log the class distribution of predictions."""
        class_counts = torch.bincount(predictions.flatten(), minlength=len(self.class_names))
        for class_idx, count in enumerate(class_counts):
            self.writer.add_scalar(f"Class_Distribution/{self.class_names[class_idx]}", count.item(), epoch)

    
    def fit(self, train_loader, val_loader, epochs, start_epoch=0):
        """Train the model."""
        evaluator = Evaluator(self.model, self.device, self.loss_fn, self.num_classes, self.metrics_config)

        # Track the best model state
        best_model_state = None
        best_metric = float("inf") if self.metric == "val_loss" else float("-inf")
        best_epoch = 0
        
        count_24=0
        count_class_24=0
        for epoch in range(start_epoch + 1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            epoch_loss, total_correct, total_pixels, total_grad_norm = 0.0, 0, 0, 0.0

            self.model.train()
            for batch_idx, (inputs, targets, _, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                unique_vals = torch.unique(inputs)
                count_24 += (targets == 24).sum().item()
                
                outputs = self.model(inputs)
                
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()

                # Compute gradient norm
                grad_norm = self.compute_gradient_norm()
                total_grad_norm += grad_norm

                self.optimizer.step()

                epoch_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)  # Shape: [batch_size, height, width]

                # Count how many pixels are predicted as class 24
                count_class_24 += (predictions == 24).sum().item()

                
                total_correct += (predictions == targets).sum().item()
                total_pixels += targets.numel()

            print("Number of pixels targets predicted as class 24(person) :",count_24)
            print(f"Number of pixels predicted as class 24(person): {count_class_24}")
            count_24=0
            count_class_24=0
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_accuracy = total_correct / total_pixels
            avg_grad_norm = total_grad_norm / len(train_loader)

            print(f"Train Loss: {avg_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Grad Norm = {avg_grad_norm:.4f}")

            # Validate the model
            val_metrics, val_loss = evaluator.evaluate(val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}")

            # Save the last checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, self.experiment_dir, is_best=False)

            # Check if this is the best model
            current_metric = val_loss if self.metric == "val_loss" else val_metrics[self.metric]
            if (self.metric == "val_loss" and current_metric < best_metric) or \
            (self.metric != "val_loss" and current_metric > best_metric):
                best_metric = current_metric
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                best_epoch = epoch

            # Early stopping logic
            if self.early_stopping_enabled:
                if current_metric == best_metric:
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1

                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered!")
                    break

            # Log metrics
            log_metrics(
                epoch=epoch,
                train_loss=avg_epoch_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_metrics=val_metrics,
                experiment_dir=self.experiment_dir,
                class_names=self.class_names
            )

            if self.tensorboard_enabled:
                # Log Training Loss and Accuracy
                self.writer.add_scalar("Loss/Training", avg_epoch_loss, epoch)
                self.writer.add_scalar("Accuracy/Training", train_accuracy, epoch)

                # Log Validation Loss and Metrics
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Validation", val_metrics.get("val_accuracy", 0.0), epoch)

                self.writer.add_scalar("Gradients/Norm", avg_grad_norm, epoch)
                self.log_learning_rate(epoch)

                if epoch % 5 == 0:  # Log every 5 epochs
                    self.log_confidence_histogram(outputs, epoch)
                self.log_class_distribution(predictions, epoch)

                # Log MeanIoU and MeanDICE
                if "MeanIoU" in val_metrics:
                    self.writer.add_scalar("Metrics/Validation/MeanIoU", val_metrics["MeanIoU"], epoch)
                if "MeanDICE" in val_metrics:
                    self.writer.add_scalar("Metrics/Validation/MeanDICE", val_metrics["MeanDICE"], epoch)
                if "IoUoverall" in val_metrics:
                    self.writer.add_scalar("Metrics/Validation/IoUoverall", val_metrics["IoUoverall"], epoch)
                print("Saving in tensorboard")
                # Log Per-Class IoU and DICE Scores
                if "IoU" in val_metrics:
                    for class_idx, iou_value in val_metrics["IoU"].items():
                        class_name = self.class_names[class_idx]
                        if class_name == "Unused":  # Skip logging for ignored classes
                            continue
                        self.writer.add_scalar(f"Metrics/Validation/IoU/{class_name}", iou_value, epoch)
                if "DICE" in val_metrics:
                    for class_idx, dice_value in val_metrics["DICE"].items():
                        class_name = self.class_names[class_idx]
                        if class_name == "Unused":  # Skip logging for ignored classes
                            continue
                        self.writer.add_scalar(f"Metrics/Validation/DICE/{class_name}", dice_value, epoch)
        # if self.tensorboard_enabled:
        #     self.log_model_info()
        # Save the best model at the end of training
        if best_model_state is not None:
            best_checkpoint_path = os.path.join(self.experiment_dir, "checkpoints", "best_checkpoint.pth")
            torch.save(best_model_state, best_checkpoint_path)
            print(f"Best model saved at epoch {best_epoch} with {self.metric}: {best_metric:.4f}")


    def validate(self, val_loader, evaluator):
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        val_loss, total_correct, total_pixels = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets, _, _ in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                val_loss += self.loss_fn(outputs, targets).item()

                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == targets).sum().item()
                total_pixels += targets.numel()

                batch_metrics = evaluator.evaluate_batch(predictions.cpu().numpy(), targets.cpu().numpy())

        val_accuracy = total_correct / total_pixels
        val_metrics = {"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy}
        val_metrics.update(batch_metrics)
        return val_loss / len(val_loader), val_metrics




class DenoisingTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir="experiments/denoising"):
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            criterion (nn.Module): Combined loss function (TotalLoss).
            optimizer (torch.optim.Optimizer): Optimizer for training.
            device (torch.device): Device for training (CPU or GPU).
            save_dir (str): Directory to save experiments.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Create experiment directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Directories for logs, checkpoints, and results
        self.log_path = os.path.join(self.experiment_dir, "logs.txt")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.results_dir = os.path.join(self.experiment_dir, "results")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.last_checkpoint_path = os.path.join(self.checkpoints_dir, "last_checkpoint.pth")
        self.best_checkpoint_path = os.path.join(self.checkpoints_dir, "best_checkpoint.pth")

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for noisy_image, clean_image in progress_bar:
            # Move input data to the correct device
            noisy_image = noisy_image.to(self.device)
            clean_image = clean_image.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(noisy_image)

            # Normalize outputs for perceptual loss
            output_norm = (output + 1) / 2
            clean_image_norm = (clean_image + 1) / 2

            # Compute losses
            total_loss, adv_loss, perc_loss, pix_loss = self.criterion.total_loss(
                output_norm, clean_image_norm, gen_output=None, real_output=None
            )

            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()

            train_loss += total_loss.item() * noisy_image.size(0)

            # Safely log losses to progress bar
            adv_loss_value = adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
            perc_loss_value = perc_loss.item() if isinstance(perc_loss, torch.Tensor) else perc_loss
            pix_loss_value = pix_loss.item() if isinstance(pix_loss, torch.Tensor) else pix_loss

            # Update progress bar with current losses
            progress_bar.set_postfix(
                total_loss=total_loss.item(),
                adv_loss=adv_loss_value,
                perc_loss=perc_loss_value,
                pix_loss=pix_loss_value
            )

        return train_loss / len(self.train_loader.dataset)

    def validate_one_epoch(self):
        """
        Validates the model for one epoch.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        val_loss = 0.0

        progress_bar = tqdm(self.val_loader, desc="Validating")

        with torch.no_grad():
            for noisy_image, clean_image in progress_bar:
                # Move input data to the correct device
                noisy_image = noisy_image.to(self.device)
                clean_image = clean_image.to(self.device)

                # Forward pass
                output = self.model(noisy_image)

                # Normalize outputs for perceptual loss
                output_norm = (output + 1) / 2
                clean_image_norm = (clean_image + 1) / 2

                # Compute losses
                total_loss, adv_loss, perc_loss, pix_loss = self.criterion.total_loss(
                    output_norm, clean_image_norm, gen_output=None, real_output=None
                )
                val_loss += total_loss.item() * noisy_image.size(0)

                # Safely log losses to progress bar
                adv_loss_value = adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
                perc_loss_value = perc_loss.item() if isinstance(perc_loss, torch.Tensor) else perc_loss
                pix_loss_value = pix_loss.item() if isinstance(pix_loss, torch.Tensor) else pix_loss

                # Update progress bar with current losses
                progress_bar.set_postfix(
                    total_loss=total_loss.item(),
                    adv_loss=adv_loss_value,
                    perc_loss=perc_loss_value,
                    pix_loss=pix_loss_value
                )

                # Save sample results for the first batch
                if len(os.listdir(self.results_dir)) < 10:  # Save up to 10 sample images
                    self.save_sample_images(noisy_image, clean_image, output)

        return val_loss / len(self.val_loader.dataset)

    def save_sample_images(self, noisy_image, clean_image, output):
        """
        Saves a few sample images (noisy, clean, and denoised).

        Args:
            noisy_image (torch.Tensor): Noisy input image.
            clean_image (torch.Tensor): Clean ground truth image.
            output (torch.Tensor): Denoised output image.
        """
        noisy_image = noisy_image[0].cpu().numpy().transpose(1, 2, 0)
        clean_image = clean_image[0].cpu().numpy().transpose(1, 2, 0)
        output = output[0].cpu().detach().numpy().transpose(1, 2, 0)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(noisy_image)
        axs[0].set_title("Noisy")
        axs[1].imshow(clean_image)
        axs[1].set_title("Clean")
        axs[2].imshow(output)
        axs[2].set_title("Denoised")
        for ax in axs:
            ax.axis("off")

        image_save_path = os.path.join(self.results_dir, f"sample_{len(os.listdir(self.results_dir)) + 1}.png")
        plt.savefig(image_save_path)
        plt.close()

    def log_metrics(self, epoch, train_loss, val_loss):
        """
        Logs metrics to a log file.

        Args:
            epoch (int): Current epoch.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
        """
        with open(self.log_path, "a") as log_file:
            log_file.write(f"Epoch [{epoch + 1}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")

    def fit(self, epochs):
        """
        Trains and validates the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
        """
        for epoch in range(epochs):
            print(f"Training started for epoch {epoch + 1}")
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            self.log_metrics(epoch, train_loss, val_loss)

            torch.save(self.model.state_dict(), self.last_checkpoint_path)  # Save last checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_checkpoint_path)  # Save best checkpoint
                print("Best model saved!")

        print("Training complete!")

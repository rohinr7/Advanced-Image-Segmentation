<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <h1>Segmentation Project: Training and Evaluation</h1>

  <h2>Note for Professor</h2>
  <p>To evaluate the best model, please follow these steps:</p>
  <ul>
    <li>Download the best model weights from the following link: <a href="https://drive.google.com/file/d/1kLjk5aNQj1XPx7fNrk7u-On7BCdmqBvR/view?usp=sharing" target="_blank">Best Model Weights</a></li>
    <li>Use the provided best configuration file: <code>best_config.yaml</code></li>
    <li>Run the following command to evaluate the model:</li>
  </ul>
  <pre><code>python evaluate.py --config ./best_config.yaml --model_path ./best_checkpoint.pth</code></pre>

  <h2>Overview</h2>
  <p>This repository contains a comprehensive pipeline for training and evaluating segmentation models using PyTorch. It is tailored for tasks such as multi-class pixel-wise segmentation in varying weather conditions, focusing on models like UNet and DeepLabV3Plus. The project also integrates an optional denoising task to enhance input data quality.</p>

  <h2>Project Structure</h2>
  <pre>
  ├── experiments/
  │   ├── configs/           # YAML or JSON configuration files for experiments
  │   ├── experiment1/       # Folder for a specific experiment
  │   │   ├── logs/          # Training logs (e.g., TensorBoard)
  │   │   ├── checkpoints/   # Saved model weights
  │   │   └── results/       # Evaluation results (e.g., metrics, images)
  ├── notebooks/
  │   ├── data_visualization.ipynb  # Dataset exploration and visualization
  │   ├── model_analysis.ipynb      # Model performance analysis
  ├── src/
  │   ├── data/
  │   │   ├── scripts/       # Data loading and preprocessing scripts
  │   │   └── dataloaders.py # Custom PyTorch DataLoader implementation
  │   ├── models/            # Model architectures
  │   │   ├── unet.py        # UNet model definition
  │   │   ├── deeplabv3plus.py # DeepLabV3Plus model definition
  │   ├── utils/             # Utility functions
  │   │   ├── metrics.py     # Evaluation metrics (IoU, accuracy, Dice Coefficient)
  │   │   ├── visualization.py # Visualization utilities
  │   │   ├── helpers.py     # Helper functions for logging, checkpointing, etc.
  │   ├── trainer.py         # Training logic and loop
  │   ├── evaluator.py       # Evaluation logic
  ├── train.py               # Main script to run training
  ├── evaluate.py            # Main script to run evaluation
  ├── requirements.txt       # Project dependencies
  ├── README.html            # Project documentation
  </pre>

  <h2>Key Features</h2>
  <ul>
      <li>Supports UNet and DeepLabV3Plus for segmentation tasks.</li>
      <li>Flexible configurations through YAML files.</li>
      <li>Denoising support as a preprocessing task.</li>
      <li>Reproducible dataset splits with train/validation/test ratios.</li>
      <li>Comprehensive metric logging (IoU, Dice Coefficient, accuracy).</li>
      <li>Data augmentation for improved generalization.</li>
  </ul>

  <h2>Installation</h2>
  <p>Clone the repository and install the dependencies:</p>
  <pre><code>git clone https://github.com/your-repository.git
cd your-repository
pip install -r requirements.txt</code></pre>

  <h2>Usage</h2>

  <h3>1. Training</h3>
  <p>Run the <code>train.py</code> script to train a model:</p>
  <pre><code>python train.py --config ./experiments/configs/config.yaml --experiment_name "Experiment_1"</code></pre>
  <p>Additional options:</p>
  <ul>
      <li><code>--resume</code>: Resume training from a checkpoint.</li>
      <li><code>--checkpoint_path</code>: Path to save/load checkpoints.</li>
  </ul>

  <h3>2. Evaluation</h3>
  <p>Run the <code>evaluate.py</code> script to evaluate a trained model:</p>
  <pre><code>python evaluate.py --config ./experiments/configs/config.yaml --model_path ./experiments/experiment1/checkpoints/model.pth</code></pre>

  <h2>Configuration</h2>
  <p>Example YAML configuration file:</p>
  <pre><code>
hyperparameters:
  batch_size: 16
  epochs: 50
  learning_rate: 0.001
  seed: 42

paths:
  data: "./data"

augmentation:
  use_augmentation: true
  params:
    RandomHorizontalFlip: 0.5
    RandomRotation: 15

dataset_split:
  train: 0.8
  val: 0.1
  test: 0.1
  </code></pre>

  <h2>Results and Logging</h2>
  <p>Training and evaluation logs are saved in the <code>experiments</code> directory. Logs include:</p>
  <ul>
      <li>Training and validation metrics per epoch.</li>
      <li>Checkpointed models and configurations.</li>
      <li>Visualization of predictions and ground truths.</li>
  </ul>

  <h2>Contributing</h2>
  <p>We welcome contributions! Feel free to submit issues or pull requests for enhancements or bug fixes.</p>

  <h2>License</h2>
  <p>Licensed under the <code>MIT License</code>.</p>
</body>
</html>

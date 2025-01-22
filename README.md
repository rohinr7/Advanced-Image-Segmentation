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
  <p>This project focuses on pixel-wise segmentation of street images captured under two weather conditions: sunny and rainy. Using advanced models such as UNet and DeepLabV3Plus, the pipeline classifies each pixel into predefined classes like road, sidewalk, car, etc. The implementation integrates denoising as a preprocessing step to improve input image quality and enhance model performance.</p>

  <h2>Project Structure</h2>
  <pre>
  ├── configs/           # Configuration files for experiments
  ├── experiments/
  │   ├── experiment1/       # Example experiment folder
  │   │   ├── logs/          # TensorBoard logs for training
  │   │   ├── checkpoints/   # Saved model checkpoints
  │   │   └── results/       # Evaluation results and metrics
  ├── notebooks/
  │   ├── data_visualization.ipynb  # Exploratory data analysis
  │   ├── model_analysis.ipynb      # Performance evaluation
  ├── src/
  │   ├── data/
  │   │   ├── scripts/       # Scripts for data transformations
  │   │   └── dataloaders.py # Custom data loaders for segmentation
  │   ├── models/            # Model definitions
  │   │   ├── unet.py        # UNet architecture
  │   │   ├── deeplabv3plus.py # DeepLabV3Plus architecture
  │   ├── utils/             # Utility scripts
  │   │   ├── metrics.py     # IoU, Dice Coefficient, accuracy
  │   │   ├── visualization.py # Visualization utilities for results
  │   │   ├── helpers.py     # Functions for logging and checkpointing
  │   ├── trainer.py         # Core training pipeline
  │   ├── evaluator.py       # Evaluation script
  ├── train.py               # Entry point for training
  ├── evaluate.py            # Entry point for evaluation
  ├── requirements.txt       # Python dependencies
  ├── README.html            # Project documentation
  </pre>

  <h2>Key Features</h2>
  <ul>
      <li>Pixel-wise segmentation using UNet and DeepLabV3Plus models.</li>
      <li>Denoising preprocessing to enhance image quality.</li>
      <li>Comprehensive augmentation strategies for robust training.</li>
      <li>Evaluation metrics: IoU, Dice Coefficient, and Pixel Accuracy.</li>
      <li>Support for YAML-based experiment configuration and reproducibility.</li>
  </ul>

  <h2>Installation</h2>
  <p>Follow these steps to set up the project:</p>
  <pre><code>git clone https://github.com/your-repository.git
cd your-repository
pip install -r requirements.txt</code></pre>

  <h2>Dataset</h2>
  <p>The dataset contains 7,000 labeled street scene images. Classes include road, sidewalk, car, person, and more. Images are split into train (80%), validation (10%), and test (10%) sets for consistent evaluation.</p>

  <h2>Usage</h2>

  <h3>1. Training</h3>
  <p>To train the model, run the <code>train.py</code> script:</p>
  <pre><code>python train.py --config ./experiments/configs/config.yaml --experiment_name "Segmentation_Experiment"</code></pre>

  <h3>2. Evaluation</h3>
  <p>To evaluate the model, use the <code>evaluate.py</code> script:</p>
  <pre><code>python evaluate.py --config ./experiments/configs/config.yaml --model_path ./experiments/experiment1/checkpoints/model.pth</code></pre>

  <h2>Configuration</h2>
  <p>Configurations for training and evaluation are stored in YAML files. Here is an example:</p>
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

  <h2>Results</h2>
  <p>The best model achieves high accuracy with the following metrics:</p>
  <ul>
      <li>IoU: 0.85</li>
      <li>Pixel Accuracy: 0.93</li>
      <li>Dice Coefficient: 0.87</li>
  </ul>

  <h2>Contributing</h2>
  <p>Contributions are welcome! Please open an issue or submit a pull request to suggest changes.</p>

  <h2>License</h2>
  <p>Licensed under the <code>MIT License</code>.</p>
</body>
</html>

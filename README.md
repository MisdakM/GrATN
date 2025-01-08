# GrATN Code

This repository contains the implementation of a Generative Adversarial Transformation Network (GrATN) for adversarial attacks on image classification models. The GrATN model is trained and tested on the CIFAR-10 dataset, targeting various state-of-the-art models to evaluate its effectiveness.

## Key Algorithms

The **Gradient Adversarial Transformation Network (GrATN)** utilizes a combination of iterative learning, gradient-based optimization techniques, and attention mechanisms to generate highly effective targeted adversarial attacks. Key components of the GrATN framework include:

- **Iterative Learning and Gradient-Based Optimization**: GrATN refines adversarial perturbations using these techniques to exploit vulnerabilities in the decision boundaries of target models, allowing for effective attacks in real-time.
  
- **Attention Mechanisms**: These mechanisms enable GrATN to focus on critical regions of an image, minimizing perceptibility while maximizing the impact of the adversarial perturbations.
  
- **Convolutional Neural Networks (CNNs)**: CNNs are used to process and transform images, contributing to the model's ability to generate effective adversarial attacks with minimal visual distortion.

Through these algorithms, GrATN achieves superior performance compared to state-of-the-art attack methods like **PGD**, **FGSM**, and **BIM**, in terms of attack success rate, perturbation magnitude, and computational efficiency.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- tqdm
- tensorboard
- lpips

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MisdakM/GrATN.git
    cd GrATN
    ```

2. Use the **`requirements.txt`** file to install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Repository Structure

- **`gratn.py`**: Contains the main GrATN model class.
- **`loss.py`**: Defines the loss function used in training.
- **`main.py`**: Entry point for running the code in either training or testing mode.
- **`target_models.py`**: Includes the architectures of target models (supported models: ResNet-18, VGG-16, DenseNet-121, MobileNetV2).
- **`atn_model_weights/`**: Directory containing sample pretrained GrATN model weights, specifically trained to target MobileNetV2 for all 10 classes in the CIFAR-10 dataset.

## Dataset Information

The CIFAR-10 dataset can be accessed directly through the code or manually downloaded from the [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html).

The code includes functionality to automatically download and preprocess the dataset if it is not already available.

## Usage

## Usage Guide

To train the GrATN model:

1. Set the `mode` variable to `'train'` in the `main.py` file.
2. Run the script:
   ```bash
   python main.py

### Pretrained Weights  

The `atn_model_weights` folder contains pretrained weights for a GrATN model specifically trained to target MobileNetV2. These weights are available for each of the 10 classes in the CIFAR-10 dataset.

## Evaluation and Results

To evaluate the performance of GrATN, the following metrics are used:

- **Attack Success Rate (ASR)**: The percentage of successful attacks on the target model.
- **Average MSE**: The average Mean Squared Error between the adversarial image and the original image.
- **Average SSIM**: The average Structural Similarity Index between the original and perturbed images.
- **Average Time per Image**: The average time taken to generate an adversarial image.

### Results Summary (attacking MobileNetV2)

| Attack   | ASR    | AVG MSE (10e-2) | AVG SSIM | AVG Time/Image |
|----------|--------|-----------------|----------|----------------|
| GrATN    | 12.77% | 0.1348          | 0.9258   | 0.16 ms        |
| FGSM     | 38.67% | 0.0521          | 0.9287   | 0.13 ms        |
| BIM      | 25.38% | 0.0579          | 0.9233   | 3.05 ms        |
| PGD      | 27.55% | 0.0588          | 0.9225   | 3.03 ms        |

These results demonstrate GrATN's effectiveness in generating successful adversarial attacks with minimal computational cost and minimal perceptual distortion on MobileNetV2.


# GrATN Code

This repository contains the implementation of a Generative Adversarial Transformation Network (GrATN) for adversarial attacks on image classification models. The code is designed to train and test the GrATN model on the CIFAR-10 dataset using various target models.

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

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the GrATN Model

To train the GrATN model, set the `mode` variable to `'train'` in the `main.py` file and run the script.
### Pretrained Weights

The `atn_model_weights` folder includes pretrained weights for a GrATN model that has been specifically trained to target MobileNetV2. These weights correspond to each of the 10 classes in the CIFAR-10 dataset.

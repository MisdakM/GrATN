import torch
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss as mse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn, optim
import lpips


class AdversarialReconstructionLossOLD(nn.Module):
    def __init__(self, targeted=True):
        super(AdversarialReconstructionLoss, self).__init__()
        self.targeted = targeted
        self.adversarial_loss = nn.CrossEntropyLoss()  # Cross-entropy loss for adversarial perturbation
        self.reconstruction_loss = nn.MSELoss()  # Mean Squared Error loss for reconstruction

    def forward(self, adv_images, adv_predictions, images, labels, target):
        # Compute adversarial loss (misclassification loss)
        if self.targeted:
            y_target = torch.zeros_like(adv_predictions)
            y_target[:, target] = 1
            adversarial_loss = self.adversarial_loss(adv_predictions, y_target)

        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_loss(adv_images, images)

        return reconstruction_loss, adversarial_loss


class AdversarialReconstructionLoss(nn.Module):
    def __init__(self, targeted=True, reconstruction='mse', device='cuda'):
        super(AdversarialReconstructionLoss, self).__init__()
        self.targeted = targeted
        self.adversarial_loss = nn.CrossEntropyLoss()  # Cross-entropy loss for adversarial perturbation
        
        if reconstruction == 'mse':
            self.reconstruction_loss = nn.MSELoss()  # Mean Squared Error loss for reconstruction
        # elif reconstruction == 'lpips':
        #     self.reconstruction_loss = lpips.LPIPS(net='alex').to(device)  # Can also use 'vgg' or 'squeeze'
        else:
            print('Unsupported Reconstruction Loss !!')
        
    def forward(self, adv_images, adv_predictions, images, labels, target=None):
        # Compute adversarial loss
        if self.targeted:
            if target is None:
                raise ValueError("Target label must be provided for targeted attacks.")
                
            # Targeted attack: maximize the probability of the target class
            y_target = torch.zeros_like(adv_predictions)
            y_target[:, target] = 1
            adversarial_loss = self.adversarial_loss(adv_predictions, y_target)
        else:
            # Untargeted attack: minimize the probability of the true label
            adversarial_loss = -self.adversarial_loss(adv_predictions, labels)

        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_loss(adv_images, images).mean()
        
        return reconstruction_loss, adversarial_loss
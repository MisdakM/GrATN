import time
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss as mse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn, optim

from utils import get_class_name

normalize = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))  # CIFAR-10 normalization values
])

class Logger:
    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        self.fn = lambda x, i: x / (i + 1) if calculate_mean else lambda x, i: x

    def __call__(self, lossX, lossXBeta, lossY, loss, metrics, i):
        track_str = f'\r{self.mode} | {i + 1:5d}/{self.length:<5d}| '
        loss_str = f'loss: {self.fn(lossXBeta, i):9.4f} +{self.fn(lossY, i):9.4f} = {self.fn(loss, i):9.4f} | '
        metric_str = ' | '.join(f'{k}: {self.fn(v, i):9.4f}' for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')

class BatchTimer:
    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()

def compute_accuracy(model, dataloader, device, ATN=None, target=None):
    model.eval()
    correct_predictions = 0
    adv_correct_predictions = 0
    total_misclassified_samples = 0
    total_misclassified_samples_into_target = 0
    total_samples = 0
    
    with tqdm(total=len(dataloader), desc=f"Computing{' Adversarial ' if ATN is not None else ' '}Accuracy", unit="batch", leave=False) as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            total_samples += labels.size(0)

            if ATN is not None:
                adv_images, _ = perturb(images, atn_model=ATN, target_model=model, target=target, device=device)
                outputs = model(normalize(images))
                adv_outputs = model(normalize(adv_images))
                _, predicted = torch.max(outputs, 1)
                _, adv_predicted = torch.max(adv_outputs, 1)
                adv_correct_predictions += (adv_predicted == predicted).sum().item()
                total_misclassified_samples += (adv_predicted != predicted).sum().item()
                misclassified_and_perturbed_to_target_idx = (predicted == labels) & (adv_predicted != predicted) & (adv_predicted == target)
                total_misclassified_samples_into_target += misclassified_and_perturbed_to_target_idx.sum().item()
            else:
                outputs = model(normalize(images))
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()

            pbar.update(1)
            
    if ATN is not None and target is not None:
        adv_accuracy = adv_correct_predictions / total_samples
        print(f'Total misclassified samples \t\t: {total_misclassified_samples}')
        print(f'Total misclassified samples into {target} \t: {total_misclassified_samples_into_target}')
        avg_target_rate = total_misclassified_samples_into_target / total_misclassified_samples if total_misclassified_samples > 0 else 0
        return adv_accuracy, avg_target_rate
    else:
        accuracy = correct_predictions / total_samples
        return accuracy

def evaluate_model(model, data_loader, device, attack=None, attack_name=None, win_size=5, show_comparison=False):
    model.eval()
    correct = 0
    total = 0
    ssim_scores = []
    mse_scores = []
    attack_times = []

    last_original_image = None
    last_perturbed_image = None
    last_original_label = None
    last_predicted_label = None

    with tqdm(total=len(data_loader), desc=f"Evaluating {attack_name or 'Clean'}", unit="batch") as pbar:
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            original_images = images.clone().detach().cpu().numpy()

            if attack:
                start_time = time.time()
                images = attack(images, labels)
                end_time = time.time()
                attack_times.extend([(end_time - start_time) / images.size(0) for _ in range(images.size(0))])
                last_original_image = original_images[-1]
                last_perturbed_image = images.detach().cpu().numpy()[-1]
                last_original_label = labels[-1].item()

            outputs = model(normalize(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if attack:
                perturbed_images = images.detach().cpu().numpy()
                for orig, pert in zip(original_images, perturbed_images):
                    ssim_scores.append(ssim(pert.transpose(1, 2, 0), orig.transpose(1, 2, 0), win_size=win_size, multichannel=True, data_range=1.0, channel_axis=-1))
                    mse_scores.append(mse(torch.tensor(orig).float(), torch.tensor(pert).float()).item() * 100)

            pbar.update(1)

    accuracy = correct / total
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_mse = np.mean(mse_scores) if mse_scores else 0.0
    avg_time_per_image = np.mean(attack_times) if attack_times else 0.0

    if show_comparison and last_original_image is not None:
        _, last_predicted_label = torch.max(model(normalize(torch.tensor(last_perturbed_image[None, ...], device=device))), 1)
        last_predicted_label = last_predicted_label.item()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(last_original_image.transpose(1, 2, 0))
        axes[0].set_title(f"Original Image (Label: {get_class_name(last_original_label)})")
        axes[1].imshow(last_perturbed_image.transpose(1, 2, 0))
        axes[1].set_title(f"Adversarial Image (Label: {get_class_name(last_predicted_label)}) - SSIM: {ssim_scores[-1]:.4f} - MSE: {mse_scores[-1]:.4f}")
        plt.suptitle(f"Attack: {attack_name}")
        plt.show()

    return accuracy, avg_ssim, avg_mse, avg_time_per_image

def cal_grad_target(X, model, target):
    x_image = X.detach()
    x_image.requires_grad_(True)
    out = model(normalize(x_image))
    target_out = out[:, target]
    target_out.backward(torch.ones_like(target_out))
    return x_image.grad

def cal_grad_untarget(X, model, y_label):
    x_image = X.detach()
    x_image.requires_grad_(True)
    y_pred = model(normalize(x_image))
    xentropy_loss_fn = nn.CrossEntropyLoss(reduce=False)
    xentropy = xentropy_loss_fn(y_pred, y_label)
    xentropy.backward(torch.ones_like(xentropy))
    return x_image.grad

def pass_atn_epoch(target_model, atn_model, criterion, atn_data, target, atn_optimizer=None, atn_scheduler=None, atn_batch_metrics={'time': BatchTimer()}, show_running=True, device='cpu', writer=None):
    mode = 'Train' if atn_model.training else 'Valid'
    logger = Logger(mode, length=len(atn_data), calculate_mean=show_running)
    total_lossX = 0
    total_lossXBeta = 0
    total_lossY = 0
    total_loss = 0
    metrics = {}

    for i_batch, (images, labels) in enumerate(atn_data):
        images = images.to(device)
        labels = labels.to(device)

        if criterion.targeted:
            images_grad = cal_grad_target(images, target_model, target)
        else:
            images_grad = cal_grad_untarget(images, target_model, labels)
        
        adv_images = atn_model(images, images_grad)
        adv_predictions = target_model(normalize(adv_images))

        lossX, lossY = criterion(adv_images, adv_predictions, images, labels, target)
        loss = lossX * atn_model.beta + lossY
        
        if atn_model.training:
            atn_optimizer.zero_grad()
            loss.backward()
            atn_optimizer.step()

        l2s = [torch.norm(images[i] - adv_images[i], p=2).item() for i in range(images.size(0))]

        metrics_batch = {metric_name: metric_fn(adv_predictions, labels).detach().cpu() for metric_name, metric_fn in atn_batch_metrics.items()}
        metrics.update({metric_name: metrics.get(metric_name, 0) + metric_batch for metric_name, metric_batch in metrics_batch.items()})
            
        if writer is not None and atn_model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1
        
        loss = loss.detach().cpu()
        total_lossX += lossX
        total_lossXBeta += lossX * atn_model.beta
        total_lossY += lossY
        total_loss += loss
        if show_running:
            logger(total_lossX, total_lossXBeta, total_lossY, total_loss, metrics, i_batch)
        else:
            logger(total_loss, metrics_batch, i_batch)
    
    if atn_model.training and atn_scheduler is not None:
        atn_scheduler.step()

    total_loss = total_loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}
            
    if writer is not None and not atn_model.training:
        writer.add_scalars('loss', {mode: total_loss.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return total_loss, metrics

def perturb(images, target_model, atn_model, target, device, labels=None):
    images = images.to(device)
    
    if target is not None:
        images_grad = cal_grad_target(images, target_model, target)
    elif labels is not None:
        labels = labels.to(device)
        images_grad = cal_grad_untarget(images, target_model, labels)
    else:
        raise ValueError("target or the labels params cannot be both None")
    
    adv_images = atn_model(images, images_grad)
    return adv_images.detach(), adv_images - images

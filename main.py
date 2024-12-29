import time
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gratn import GrATN
from loss import AdversarialReconstructionLoss
from training import BatchTimer, accuracy, compute_accuracy, evaluate_model, pass_atn_epoch, perturb
from utils import download_weights, get_class_name
from target_models import resnet18, densenet121, vgg16_bn, mobilenet_v2


def main(mode='train'):
    device = get_device()
    train_loader, test_loader = get_data_loaders()
    model = load_model(device)

    atn_model = GrATN(beta=2000).to(device)
    if torch.cuda.device_count() > 1:
        print('Using more than one GPU')
        atn_model = nn.DataParallel(atn_model)

    target = 0

    if mode == 'train':
        atn_epochs = 3
        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        atn_model, atn_criterion, atn_optimizer, atn_scheduler, atn_metrics, best_atn_model_path, accuracy_tolerance = initialize_atn(
            device, atn_model)

        train_atn(device, model, atn_model, atn_criterion, atn_optimizer, atn_scheduler, atn_metrics,
                  atn_epochs, target, best_atn_model_path, accuracy_tolerance, train_loader, test_loader, writer)

    elif mode == 'test':
        # best_model_weights = torch.load('atn_model_weights/best_atn_model_weights.pth')
        best_model_weights = torch.load(
            'atn_model_weights/0-target-0.9391-0.1966-2000-15epochs-0.1135mse-0.9435-ConvAtten.pth')
        atn_model.load_state_dict(best_model_weights)
        print("Best model weights loaded successfully.")

        data_loader = test_loader

        print('-'*50)

        accuracy_before_atn = compute_accuracy(model, data_loader, device)
        accuracy_after_atn, avg_target_rate = compute_accuracy(
            model, data_loader, device, ATN=atn_model, target=target)

        print('-'*50)
        print(f'Accuracy Drop \t\t\t: {accuracy_before_atn * 100:.2f}% -> {accuracy_after_atn * 100:.2f}%')
        print(f'Avg Target Rate for {get_class_name(target)}\t: {avg_target_rate * 100:.2f}%')  # targeted_predictions / total_samples
        print('-'*50)
        print(f'Accuracy drop due to ATN\t: {(accuracy_before_atn - accuracy_after_atn) * 100:.2f}%')


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader


def load_model(device, model_name='mobilenet_v2'):
    download_weights()

    if model_name == 'resnet18':
        model = resnet18(num_classes=10).to(device)
        state_dict = torch.load(
            './cifar10_models/state_dicts/resnet18.pt', map_location=device)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=10).to(device)
        state_dict = torch.load(
            './cifar10_models/state_dicts/densenet121.pt', map_location=device)
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(num_classes=10).to(device)
        state_dict = torch.load(
            './cifar10_models/state_dicts/vgg16_bn.pt', map_location=device)
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=10).to(device)
        state_dict = torch.load(
            './cifar10_models/state_dicts/mobilenet_v2.pt', map_location=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(state_dict)
    print(f"{model_name} model weights loaded successfully.")
    return model


def initialize_atn(device, atn_model):
    atn_criterion = AdversarialReconstructionLoss(device=device)
    atn_optimizer = torch.optim.AdamW(
        atn_model.parameters(), lr=1e-4, weight_decay=1e-4)
    atn_scheduler = None
    atn_metrics = {'fps': BatchTimer(), 'acc': accuracy}
    best_atn_model_path = 'atn_model_weights/best_atn_model_weights.pth'
    accuracy_tolerance = 1e-2
    return atn_model, atn_criterion, atn_optimizer, atn_scheduler, atn_metrics, best_atn_model_path, accuracy_tolerance


def train_atn(device, model, atn_model, atn_criterion, atn_optimizer, atn_scheduler, atn_metrics, atn_epochs, target, best_atn_model_path, accuracy_tolerance, train_loader, test_loader, writer):
    atn_train_losses, atn_train_accuracies = [], []
    atn_val_losses, atn_val_accuracies, atn_val_ssims = [], [], []
    lowest_model_accuracy, best_val_ssim = float('inf'), 0.0

    if atn_criterion.targeted:
        print(f"Targeting : {target} -> {get_class_name(target)}")
    else:
        print("Performing untargeted Attacks...")

    start_time = time.time()
    for epoch in range(atn_epochs):
        print(f'\nATN Epoch {epoch + 1}/{atn_epochs}\n{"-" * 10}')
        atn_model.train()
        atn_avg_loss, atn_epoch_metrics = pass_atn_epoch(
            model, atn_model, atn_criterion, train_loader, target, atn_optimizer, atn_scheduler,
            atn_batch_metrics=atn_metrics, show_running=True, device=device, writer=writer
        )
        atn_train_accuracies.append(atn_epoch_metrics['acc'].item())
        atn_train_losses.append(atn_avg_loss.item())

        atn_model.eval()
        atn_val_accuracy, atn_val_avg_ssim, atn_val_avg_mse, atn_val_avg_time_per_image = evaluate_model(
            model, test_loader, device,
            attack=lambda x, y: perturb(
                x, model, atn_model, target, device=device)[0],
            attack_name="GrATN (Validation)", show_comparison=True
        )
        atn_val_accuracies.append(atn_val_accuracy)
        atn_val_losses.append(0)  # TODO: Compute Loss for Val Set
        atn_val_ssims.append(atn_val_avg_ssim)
        print(f'Validation: Accuracy: {atn_val_accuracy:.4f} | Avg. SSIM: {atn_val_avg_ssim:.4f} | Avg. Time/Image: {atn_val_avg_time_per_image * 1000:.2f} ms')

        if atn_val_accuracy < lowest_model_accuracy:
            lowest_model_accuracy = atn_val_accuracy
            best_val_ssim = atn_val_avg_ssim
            torch.save(atn_model.state_dict(), best_atn_model_path)
            print('--' * 50)
            print(f'--> Saving best GrATN model with accuracy (tolerance: {accuracy_tolerance}): {lowest_model_accuracy:.4f} and SSIM: {best_val_ssim:.4f}')
            print('--' * 50)

    training_time = time.time() - start_time
    print(f"Training complete in {training_time //60:.0f}m {training_time % 60:.0f}s")
    print('Done training ATN.')


if __name__ == "__main__":
    # Default execution: training mode
    main()

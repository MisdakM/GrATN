
import os
import zipfile
import requests
import torch
import gc
from tqdm import tqdm

# Clean up unused variables and tensors


def clean_up_memory():
    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()  # Garbage collect Python objects


def unnormalize_cifar10(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1)

    tensor = tensor * std + mean
    return tensor


def get_class_name(class_number):
    cifar10_classes = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if 0 <= class_number < len(cifar10_classes):
        return cifar10_classes[class_number]
    else:
        raise ValueError(
            f"Invalid CIFAR-10 class number: {class_number}. Must be between 0 and 9.")


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


# Clean up unused variables and tensors
def clean_up_memory():
    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()  # Garbage collect Python objects


def unnormalize_cifar10(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    return tensor


def get_class_name(class_number):
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if 0 <= class_number < len(cifar10_classes):
        return cifar10_classes[class_number]
    else:
        raise ValueError(f"Invalid CIFAR-10 class number: {class_number}. Must be between 0 and 9.")


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def download_weights():
    url = (
        "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    )
    temp_dir = os.path.join(os.getcwd(), "temp")
    zip_file_path = os.path.join(temp_dir, "state_dicts.zip")
    extract_dir = os.path.join(os.getcwd(), "cifar10_models")

    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Check if the zip file already exists
    if os.path.exists(zip_file_path):
        print("Zip file already exists. Skipping download.")
    else:
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open(zip_file_path, "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful.")

    # Check if the extraction directory already exists
    if os.path.exists(extract_dir):
        print("Files already extracted. Skipping extraction.")
    else:
        print("Unzipping file...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            print("Unzip file successful!")
        
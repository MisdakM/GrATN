
import matplotlib.pyplot as plt
import torch
import random

from utils import unnormalize_cifar10


def visualize_image(image_tensor, title, unnormalize=False):
    if unnormalize:
        image_tensor = unnormalize_cifar10(image_tensor)
    
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Display the image using matplotlib
    plt.imshow(image_np)
    plt.title(f"{title}")
    plt.show()
    
    
def visualize_images(image_tensors, titles, unnormalize=False):
    fig, axes = plt.subplots(1, len(image_tensors), figsize=(15, 5))  # Create a figure with three subplots
    
    for i, (image_tensor, title) in enumerate(zip(image_tensors, titles)):
        
        if unnormalize:
            image_tensor = unnormalize_cifar10(image_tensor)
        
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Display the image using matplotlib
        axes[i].imshow(image_np)
        axes[i].set_title(title)
#         axes[i].set_title(f'[{torch.min(img):.2f}, {torch.max(img):.2f}] {torch.mean(img):.2f}')
        axes[i].axis('off')

    plt.show()
    
    
def visualize_random_image(batch):
    sample_imgs, sample_labels = batch
    index = int(random.uniform(0, sample_imgs.size(0) - 1))    
    
    img = sample_imgs[index]
    lbl = sample_labels[index]

    visualize_image(img, str(lbl.item()))
    return img, lbl

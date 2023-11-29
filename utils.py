import os
import random
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image

def convolution2d(input_tensor, custom_kernel, stride, padding=0):
    # input_tensor : [1, c, image_size, image_size]
    # custom_kernel : [kernel_size, kernel_size]
    # stride : convolution stride 
    # perform convolution on every channel with custom kernel
    kernel_size = custom_kernel.size(0)
    output_channels = []

    # Loop through each input channel and apply the custom convolution
    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].unsqueeze(1)  # Select the channel
        output_channel = F.conv2d(input_channel, custom_kernel.view(1, 1, kernel_size, kernel_size), stride=stride, padding=padding)
        output_channels.append(output_channel)

    # Stack the output channels along the channel dimension
    return torch.cat(output_channels, dim=1)


def split_data(source_folder, target_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None):
    """
    Split data into training, validation, and test sets.

    Parameters:
        source_folder (str): Path to the source folder containing images.
        target_folder (str): Path to the target folder to save the split sets.
        train_ratio (float): Ratio of training set (default is 0.7).
        val_ratio (float): Ratio of validation set (default is 0.2).
        test_ratio (float): Ratio of test set (default is 0.1).
        seed (int): Seed for reproducibility (default is None).
    """
    # assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Create destination folders
    train_folder = os.path.join(target_folder, 'train', 'hq')
    val_folder = os.path.join(target_folder, 'val', 'hq')
    test_folder = os.path.join(target_folder, 'test', 'hq')

    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # List all files in the source folder
    all_files = os.listdir(source_folder)
    # Shuffle the list of files
    random.shuffle(all_files)

    # Calculate the number of files for each set
    num_files = len(all_files)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)

    # Assign files to sets
    train_set = all_files[:num_train]
    val_set = all_files[num_train:num_train + num_val]
    test_set = all_files[num_train + num_val:]

    # Copy files to destination folders
    for file in train_set:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

    for file in val_set:
        shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))

    for file in test_set:
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

# # Example usage:
# source_folder = '/eva_data2/shlu2240/Dataset/celeba_hq_256_raw'
# target_folder = '/eva_data2/shlu2240/Dataset/celeba_hq'
# split_data(source_folder, target_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42)

# given hq, kernel. Return lr using convolution
def apply_kernel_downsample(hq_path, kernel, scale):
    # Load the high-quality image using PIL
    hq_image = Image.open(hq_path).convert("RGB")

    # Convert PIL image to PyTorch tensor
    hq_tensor = to_tensor(hq_image).unsqueeze(0)

    # Convert the kernel to a PyTorch tensor
    kernel_tensor = torch.FloatTensor(kernel)

    # Apply the kernel using convolution
    output_tensor = convolution2d(hq_tensor, kernel_tensor, padding=kernel.shape[0] // 2, stride=scale)

    # Convert the result back to a PIL image
    downscaled_image = to_pil_image(output_tensor.squeeze(0))

    return downscaled_image

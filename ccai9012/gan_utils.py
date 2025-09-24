"""
GAN Utilities Module
===================

This module provides comprehensive utilities for training and using Generative Adversarial Networks (GANs),
specifically focused on image-to-image translation tasks with Pix2Pix-style architectures.

The module covers the entire GAN workflow from data preparation to model training and inference,
making it easy to implement image translation tasks between paired domains.

Main components:
- Data preparation: Functions to organize and process image pairs
- Data loading: Custom datasets and dataloaders for paired images
- Model architectures: U-Net generator and PatchGAN discriminator implementations
- Training pipeline: Complete GAN training workflow with appropriate losses
- Inference utilities: Functions for using trained models and visualizing results

Usage:
    ### Prepare dataset
    train_count, test_count = prepare_gan_dataset(source_root="source_data",
                                                 train_root="data/train",
                                                 test_root="data/test")

    ### Create data loaders
    train_loader = create_paired_data_loader("data/train", batch_size=4)

    ### Initialize models
    G = UNetGenerator()
    D = PatchDiscriminator()

    ### Train GAN
    G, history = train_GAN(G, D, train_loader, num_epochs=100)

    ### Inference
    fake_img = inference_gan(G, "data/test/A", "results/")
"""

import os
import random
import glob
from PIL import Image, ImageOps
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# =================================================================
# Part 1: Data Preparation and Processing
# Functions for preparing and organizing the dataset structure
# =================================================================

def setup_directories(base_path: str = "data") -> None:
    """
    Create necessary directory structure for GAN training.

    Args:
        base_path: Base directory path for data
    """
    for split in ['train', 'test']:
        for folder in ['A', 'B']:
            os.makedirs(os.path.join(base_path, split, folder), exist_ok=True)

def collect_image_pairs(source_root: str) -> List[Tuple[str, str, str, str]]:
    """
    Collect paired source and target images from the source directory.

    Args:
        source_root: Root directory containing source data

    Returns:
        List of tuples (region, relative_path, source_path, target_path)
    """
    pairs = []
    valid_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for region in os.listdir(source_root):
        region_path = os.path.join(source_root, region)
        source_dir = os.path.join(region_path, "Source")
        target_dir = os.path.join(region_path, "Target")

        for root, _, files in os.walk(source_dir):
            for file in files:
                if not any(file.lower().endswith(ext) for ext in valid_exts):
                    continue
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                if os.path.exists(target_path):
                    pairs.append((region, relative_path, source_path, target_path))

    return pairs

def split_pairs(pairs: List[Tuple], train_ratio: float = 0.85, random_seed: int = 42) -> Tuple[List, List]:
    """
    Split image pairs into training and testing sets.

    Args:
        pairs: List of image pairs
        train_ratio: Ratio of training set
        random_seed: Random seed for reproducibility

    Returns:
        (train_pairs, test_pairs)
    """
    random.seed(random_seed)
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)
    num_train = int(len(pairs_copy) * train_ratio)
    return pairs_copy[:num_train], pairs_copy[num_train:]

def process_and_save_image(image_path: str, dst_path: str) -> None:
    """
    Process and save a single image (handles RGBA/LA formats).

    Args:
        image_path: Source image path
        dst_path: Destination save path
    """
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'LA':
            alpha = img.split()[1]
        else:
            alpha = img.split()[3]
        background.paste(img.convert('RGB'), mask=alpha)
        img = background
    img.save(dst_path)

def copy_pair(pair_list: List[Tuple], split_root: str) -> None:
    """
    Copy and process image pairs to train/test directories.

    Args:
        pair_list: List of image pairs
        split_root: Destination root directory
    """
    for region, rel_path, source_path, target_path in pair_list:
        parts = rel_path.split(os.sep)
        new_name = f"{region}_{'_'.join(parts)}"

        dst_A = os.path.join(split_root, "A", new_name)
        dst_B = os.path.join(split_root, "B", new_name)

        process_and_save_image(source_path, dst_A)
        process_and_save_image(target_path, dst_B)

def prepare_gan_dataset(source_root: str = "data/Exp4",
                       train_root: str = "data/train",
                       test_root: str = "data/test",
                       train_ratio: float = 0.85) -> Tuple[int, int]:
    """
    Main function to prepare GAN training dataset.
    Creates directories, collects images, splits data, and processes images.

    Args:
        source_root: Source data root directory
        train_root: Training set directory
        test_root: Testing set directory
        train_ratio: Training set ratio

    Returns:
        (train_count, test_count): Number of samples in training and testing sets
    """
    # Create directories
    setup_directories(os.path.dirname(train_root))

    # Collect image pairs
    pairs = collect_image_pairs(source_root)

    # Split dataset
    train_pairs, test_pairs = split_pairs(pairs, train_ratio)

    # Copy and process images
    copy_pair(train_pairs, train_root)
    copy_pair(test_pairs, test_root)

    return len(train_pairs), len(test_pairs)

# =================================================================
# Part 2: Data Loading and Augmentation
# Dataset and DataLoader classes for training
# =================================================================

def augment_pair(image_A: Image.Image,
                image_B: Image.Image,
                flip_prob: float = 0.5,
                rotate_prob: float = 0.3,
                max_rotation: int = 30,
                brightness: float = 0.2,
                contrast: float = 0.2) -> Tuple[Image.Image, Image.Image]:
    """
    Apply synchronized data augmentation to a pair of images.
    Ensures both input and target images undergo the same transformations.

    Args:
        image_A: Input image (Source)
        image_B: Target image (Target)
        flip_prob: Probability of horizontal flip
        rotate_prob: Probability of random rotation
        max_rotation: Maximum rotation degree (+/-)
        brightness: Max brightness adjustment factor
        contrast: Max contrast adjustment factor

    Returns:
        Tuple of augmented images (aug_A, aug_B)
    """

    # --------------------------
    # Horizontal flip
    # --------------------------
    if random.random() < flip_prob:
        image_A = ImageOps.mirror(image_A)
        image_B = ImageOps.mirror(image_B)

    # --------------------------
    # Random rotation
    # --------------------------
    if random.random() < rotate_prob:
        angle = random.uniform(-max_rotation, max_rotation)
        image_A = image_A.rotate(angle, resample=Image.BILINEAR)
        image_B = image_B.rotate(angle, resample=Image.BILINEAR)

    # --------------------------
    # Brightness adjustment
    # --------------------------
    if brightness > 0:
        factor = random.uniform(1 - brightness, 1 + brightness)
        image_A = TF.adjust_brightness(image_A, factor)
        image_B = TF.adjust_brightness(image_B, factor)

    # --------------------------
    # Contrast adjustment
    # --------------------------
    if contrast > 0:
        factor = random.uniform(1 - contrast, 1 + contrast)
        image_A = TF.adjust_contrast(image_A, factor)
        image_B = TF.adjust_contrast(image_B, factor)

    return image_A, image_B


class PairedImageDataset(Dataset):
    """
    Custom Dataset class for paired images training.
    Loads corresponding images from A/ and B/ directories.
    Assumes both directories have matching filenames.
    """

    def __init__(self, root_dir, transform=None):
        self.A_paths = sorted(glob.glob(os.path.join(root_dir, 'A', '*.png')))
        self.B_paths = sorted(glob.glob(os.path.join(root_dir, 'B', '*.png')))
        assert len(self.A_paths) == len(self.B_paths), "A/B image counts must match"
        self.transform = transform

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, idx):
        A_img = Image.open(self.A_paths[idx]).convert('RGB')
        B_img = Image.open(self.B_paths[idx]).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img}

def create_paired_data_loader(data_dir: str, batch_size: int = 32) -> DataLoader:
    """
    Creates a DataLoader with standard GAN image transformations.

    Args:
        data_dir: Directory containing paired images (with A/ and B/ subdirs)
        batch_size: Number of samples per batch

    Returns:
        DataLoader object configured for GAN training
    """
    # Define standard transformations
    transform_list = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Create transformation pipeline
    transform = transforms.Compose(transform_list)

    # Create dataset
    dataset = PairedImageDataset(data_dir, transform=transform)

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader

# =================================================================
# Part 3: Model Definition
# Neural network architectures for Generator and Discriminator
# =================================================================

class UNetGenerator(nn.Module):
    # Simple U-Net for demo purposes
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(features, features * 2, 4, 2, 1), nn.BatchNorm2d(features * 2),
                                   nn.LeakyReLU(0.2))
        self.down3 = nn.Sequential(nn.Conv2d(features * 2, features * 4, 4, 2, 1), nn.BatchNorm2d(features * 4),
                                   nn.LeakyReLU(0.2))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1), nn.BatchNorm2d(features * 2),
                                 nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(features * 2, features, 4, 2, 1), nn.BatchNorm2d(features),
                                 nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(features, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(d3)
        u2 = self.up2(u1 + d2)  # skip connection
        u3 = self.up3(u2 + d1)
        return u3


# Define Discriminator (PatchGAN)
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, 1, 4, 1, 1)  # output 1-channel patch
        )

    def forward(self, x):
        return self.model(x)

# =================================================================
# Part 4: Training Functions
# Loss functions and training step implementation
# =================================================================

def train_GAN(G, D, train_loader, num_epochs=50, log_interval=10,
              save_dir='checkpoints', save_interval=20,
              device=None, lambda_L1=100, lr=0.0002, betas=(0.5, 0.999)):
    """
    Train a Pix2Pix GAN model with given Generator and Discriminator.
    Losses, optimizers, and optional model checkpoint saving are handled internally.

    Parameters:
        G (nn.Module)             -- Pre-initialized Generator network
        D (nn.Module)             -- Pre-initialized Discriminator network
        train_loader (DataLoader) -- PyTorch DataLoader providing training data pairs (A, B)
        num_epochs (int)          -- Number of training epochs
        log_interval (int)        -- Steps between printing loss logs
        save_dir (str or None)    -- Directory to save model checkpoints. If None, checkpoints are not saved
        save_interval (int)       -- Save model every `save_interval` epochs
        device (str or None)      -- 'cuda', 'cpu', or None for auto selection
        lambda_L1 (float)         -- Weight for L1 loss relative to GAN loss
        lr (float)                -- Learning rate for Adam optimizer
        betas (tuple)             -- Beta parameters for Adam optimizer

    Returns:
        G (nn.Module)  -- Trained generator network
        history (dict) -- Dictionary containing lists of discriminator and generator losses:
                          history['loss_D'], history['loss_G']
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    history = {'loss_D': [], 'loss_G': []}

    G.to(device)
    D.to(device)
    G.train()
    D.train()

    # --- Loss functions ---
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    # --- Optimizers ---
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            fake_B = G(real_A)
            real_AB = torch.cat([real_A, real_B], 1)
            fake_AB = torch.cat([real_A, fake_B.detach()], 1)

            D_real_out = D(real_AB)
            D_fake_out = D(fake_AB)
            real_label = torch.ones_like(D_real_out)
            fake_label = torch.zeros_like(D_fake_out)

            loss_D_real = criterion_GAN(D_real_out, real_label)
            loss_D_fake = criterion_GAN(D_fake_out, fake_label)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            fake_AB = torch.cat([real_A, fake_B], 1)
            D_fake_out = D(fake_AB)
            real_label_G = torch.ones_like(D_fake_out)

            loss_G_GAN = criterion_GAN(D_fake_out, real_label_G)
            loss_G_L1 = criterion_L1(fake_B, real_B) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # --- Logging ---
            if i % log_interval == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Step [{i}/{len(train_loader)}] "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

            history['loss_D'].append(loss_D.item())
            history['loss_G'].append(loss_G.item())

        # --- Save model at intervals ---
        if save_dir is not None and ((epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs):
            torch.save(G.state_dict(), os.path.join(save_dir, f'G_epoch_{epoch + 1}.pth'))
            torch.save(D.state_dict(), os.path.join(save_dir, f'D_epoch_{epoch + 1}.pth'))
            print(f"Models saved at epoch {epoch + 1}")

    return G, history

# =================================================================
# Part 5: Inference Functions
# Utilities for model inference and visualization
# =================================================================

@torch.no_grad()
# Helper function to convert tensor to image
def tensor2img(t):
    t = t.cpu().squeeze(0)
    t = (t + 1) / 2.0  # [-1,1] -> [0,1]
    t = t.permute(1, 2, 0).numpy()
    t = np.clip(t, 0, 1)
    t = (t * 255).astype(np.uint8)
    return t


def inference_gan(G, test_A_dir, results_dir='results/', device=None):
    """
    Run inference with a trained Pix2Pix generator on a folder of test images.

    Parameters:
        G (nn.Module)        -- trained generator
        test_A_dir (str)     -- directory of input images (A)
        results_dir (str)    -- directory to save generated images
        device (str or torch.device, optional) -- 'cuda' or 'cpu'; if None, automatically select

    Returns:
        None (saves images in results_dir)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    G.to(device)
    G.eval()
    os.makedirs(results_dir, exist_ok=True)

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Iterate over test images
    for img_name in sorted(os.listdir(test_A_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        A_path = os.path.join(test_A_dir, img_name)
        A_img = Image.open(A_path).convert("RGB")
        A_tensor = transform(A_img).unsqueeze(0).to(device)

        with torch.no_grad():
            fake_B = G(A_tensor)

        # Convert to image and save
        fake_B_img = Image.fromarray(tensor2img(fake_B))
        fake_B_img.save(os.path.join(results_dir, img_name))

    print("Inference done! Results saved to", results_dir)

    return fake_B_img

def load_model(G, model_path, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    G.to(device)

    return G

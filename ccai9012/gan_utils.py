"""Utilities for paired image-to-image translation with Pix2Pix-style GANs.

The module covers dataset preparation, synchronized augmentation, data loading,
compact teaching models, training, checkpoint loading, and folder-based
inference.

Example:
    Prepare a paired dataset and create its training loader::

        train_count, test_count = prepare_gan_dataset("source_data")
        train_loader = create_paired_data_loader("data/train", batch_size=4)
"""

from __future__ import annotations

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
    """Create ``train/A``, ``train/B``, ``test/A``, and ``test/B`` folders.

    Args:
        base_path: Parent directory in which the train and test folders are
            created. Existing folders are preserved.

    Returns:
        None.
    """
    for split in ['train', 'test']:
        for folder in ['A', 'B']:
            os.makedirs(os.path.join(base_path, split, folder), exist_ok=True)

def collect_image_pairs(source_root: str) -> List[Tuple[str, str, str, str]]:
    """Find source images that have a target image at the matching path.

    Args:
        source_root: Directory containing region folders. Each region must
            contain ``Source`` and ``Target`` subdirectories.

    Returns:
        Tuples containing the region name, relative image path, source path,
        and matching target path. Source images without a target are omitted.
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

def split_pairs(
    pairs: List[Tuple], train_ratio: float = 0.85, random_seed: int = 42
) -> Tuple[List, List]:
    """Shuffle image pairs reproducibly and split them into two lists.

    Args:
        pairs: Image-pair records such as those returned by
            :func:`collect_image_pairs`.
        train_ratio: Fraction assigned to the training list.
        random_seed: Seed used for the local split order.

    Returns:
        The training pairs followed by the remaining test pairs.
    """
    random.seed(random_seed)
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)
    num_train = int(len(pairs_copy) * train_ratio)
    return pairs_copy[:num_train], pairs_copy[num_train:]

def process_and_save_image(image_path: str, dst_path: str) -> None:
    """Save an image, compositing transparent pixels over white when needed.

    Args:
        image_path: Path to the source image.
        dst_path: Path at which the processed image is saved.

    Returns:
        None.
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
    """Copy paired images into matching ``A`` and ``B`` folders.

    Args:
        pair_list: Pair records containing region, relative path, source path,
            and target path.
        split_root: Destination split directory, such as ``data/train``.

    Returns:
        None.
    """
    # Ensure destination directories exist even if the caller didn't run setup_directories
    os.makedirs(os.path.join(split_root, "A"), exist_ok=True)
    os.makedirs(os.path.join(split_root, "B"), exist_ok=True)

    for region, rel_path, source_path, target_path in pair_list:
        parts = rel_path.split(os.sep)
        new_name = f"{region}_{'_'.join(parts)}"

        dst_A = os.path.join(split_root, "A", new_name)
        dst_B = os.path.join(split_root, "B", new_name)

        # Be defensive: make sure parent dirs exist right before saving
        os.makedirs(os.path.dirname(dst_A), exist_ok=True)
        os.makedirs(os.path.dirname(dst_B), exist_ok=True)

        process_and_save_image(source_path, dst_A)
        process_and_save_image(target_path, dst_B)

def prepare_gan_dataset(source_root: str = "data/Exp4",
                       train_root: str = "data/train",
                       test_root: str = "data/test",
                       train_ratio: float = 0.85) -> Tuple[int, int]:
    """Prepare train and test folders from a region-based paired dataset.

    This convenience function collects matching images, splits the pairs, and
    copies processed images into ``A`` and ``B`` folders.

    Args:
        source_root: Root containing region folders with ``Source`` and
            ``Target`` subdirectories.
        train_root: Destination directory for training pairs.
        test_root: Destination directory for test pairs.
        train_ratio: Fraction of pairs assigned to training.

    Returns:
        Number of training pairs and number of test pairs.
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
    """Apply the same random augmentation parameters to two paired images.

    Args:
        image_A: Source-domain image.
        image_B: Target-domain image.
        flip_prob: Probability of a horizontal flip.
        rotate_prob: Probability of rotation.
        max_rotation: Maximum absolute rotation angle in degrees.
        brightness: Maximum brightness-factor deviation from 1.
        contrast: Maximum contrast-factor deviation from 1.

    Returns:
        The augmented source and target images.
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
    """PyTorch dataset that loads matching images from ``A`` and ``B``.

    Args:
        root_dir: Directory containing ``A`` and ``B`` subdirectories.
        transform: Optional callable applied independently to both images.

    Raises:
        AssertionError: If the two directories contain different image counts.

    Note:
        Images are paired by their sorted path order. Use matching filenames in
        both folders to avoid accidental misalignment.
    """

    def __init__(self, root_dir: str, transform=None) -> None:
        self.A_paths = sorted(glob.glob(os.path.join(root_dir, 'A', '*.png')))
        self.B_paths = sorted(glob.glob(os.path.join(root_dir, 'B', '*.png')))
        assert len(self.A_paths) == len(self.B_paths), "A/B image counts must match"
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of image pairs."""
        return len(self.A_paths)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Load and transform one source-target image pair.

        Args:
            idx: Zero-based pair index.

        Returns:
            A mapping with the transformed source under ``A`` and target under
            ``B``.
        """
        A_img = Image.open(self.A_paths[idx]).convert('RGB')
        B_img = Image.open(self.B_paths[idx]).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img}

def create_paired_data_loader(data_dir: str, batch_size: int = 32) -> DataLoader:
    """Create a shuffled data loader for normalized 256 × 256 image pairs.

    Args:
        data_dir: Directory containing paired ``A`` and ``B`` subdirectories.
        batch_size: Number of image pairs per batch.

    Returns:
        A data loader with resize, tensor conversion, and ``[-1, 1]``
        normalization transforms.
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
    """Compact encoder-decoder generator with additive skip connections.

    Args:
        in_channels: Number of channels in each source image.
        out_channels: Number of channels generated for each output image.
        features: Channel count in the first encoder block.

    Note:
        This is a small teaching architecture rather than a full production
        U-Net.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a target-domain batch from a source-domain batch.

        Args:
            x: Tensor shaped ``(batch, in_channels, height, width)``.

        Returns:
            Generated tensor with values bounded by the final ``tanh`` layer.
        """
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(d3)
        u2 = self.up2(u1 + d2)  # skip connection
        u3 = self.up3(u2 + d1)
        return u3


# Define Discriminator (PatchGAN)
class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for concatenated source-target image pairs.

    Args:
        in_channels: Combined channel count of the source and target images.
        features: Channel count in the first convolutional block.
    """

    def __init__(self, in_channels: int = 6, features: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, 1, 4, 1, 1)  # output 1-channel patch
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score local patches in a concatenated image pair.

        Args:
            x: Concatenated source-target tensor.

        Returns:
            A one-channel grid of patch scores.
        """
        return self.model(x)

# =================================================================
# Part 4: Training Functions
# Loss functions and training step implementation
# =================================================================

def train_GAN(
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    log_interval: int = 10,
    save_dir: str | None = 'checkpoints',
    save_interval: int = 20,
    device: str | torch.device | None = None,
    lambda_L1: float = 100,
    lr: float = 0.0002,
    betas: tuple[float, float] = (0.5, 0.999),
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a generator and discriminator with Pix2Pix losses.

    Args:
        G: Initialized generator network.
        D: Initialized discriminator network.
        train_loader: Batches containing source tensor ``A`` and target tensor
            ``B``.
        num_epochs: Number of complete passes through the loader.
        log_interval: Batch interval between printed loss messages.
        save_dir: Checkpoint directory, or ``None`` to disable saving.
        save_interval: Epoch interval between checkpoints.
        device: PyTorch device or device name. ``None`` selects CUDA when
            available and otherwise uses CPU.
        lambda_L1: Weight of the pixel-wise L1 term.
        lr: Adam learning rate for both networks.
        betas: Adam beta coefficients.

    Returns:
        The trained generator and a history containing ``loss_D`` and
        ``loss_G`` values for every batch.
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

def tensor2img(t: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor into an 8-bit RGB array.

    Args:
        t: Tensor with values expected in ``[-1, 1]``. A leading singleton
            batch dimension is removed.

    Returns:
        NumPy array shaped ``(height, width, channels)`` with ``uint8`` values.
    """
    with torch.no_grad():
        t = t.cpu().squeeze(0)
        t = (t + 1) / 2.0  # [-1,1] -> [0,1]
        t = t.permute(1, 2, 0).numpy()
        t = np.clip(t, 0, 1)
        t = (t * 255).astype(np.uint8)
        return t


def inference_gan(
    G: nn.Module,
    test_A_dir: str,
    results_dir: str = 'results/',
    device: str | torch.device | None = None,
) -> Image.Image:
    """Generate and save target images for every supported image in a folder.

    Args:
        G: Trained generator network.
        test_A_dir: Directory containing source-domain PNG or JPEG images.
        results_dir: Directory in which generated images are saved.
        device: PyTorch device or device name. ``None`` selects CUDA when
            available and otherwise uses CPU.

    Returns:
        The last generated image. All generated images are also written to
        ``results_dir`` with their input filenames.

    Raises:
        UnboundLocalError: If ``test_A_dir`` contains no supported image files.
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

def load_model(
    G: nn.Module,
    model_path: str,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Load generator weights and prepare the model for inference.

    Args:
        G: Initialized generator architecture compatible with the checkpoint.
        model_path: Path to a saved PyTorch state dictionary.
        device: PyTorch device or device name. ``None`` selects CUDA when
            available and otherwise uses CPU.

    Returns:
        The supplied generator in evaluation mode on the selected device.

    Raises:
        RuntimeError: If the checkpoint is incompatible with the generator.
        FileNotFoundError: If ``model_path`` does not exist.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    G.to(device)

    return G

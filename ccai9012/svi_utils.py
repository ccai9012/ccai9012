"""
Street View Imagery (SVI) Utilities
===================================

This module provides tools for working with Google Street View Images (SVIs), including downloading,
segmentation, and visualization functionalities. It enables users to download street view imagery
using Google Maps API, perform semantic segmentation on these images using pre-trained models,
and visualize the segmentation results.

The module is organized into several components:
- Google Street View downloader: Tools for fetching street view images using the Google Maps API
- Segmentation utilities: Functions to perform semantic segmentation on street view images
- Visualization utilities: Functions to visualize original images alongside their segmentations

This is particularly useful for urban analysis, streetscape assessment, and understanding
the composition of street-level imagery through semantic segmentation.

"""

import os
import numpy as np
import torch
import getpass
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

CITYSCAPES_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

# === Pulling SVIs from Google Map ===
class GoogleSVIDownloader:
    """
    A utility class for downloading Google Street View Images (SVIs) using Google Maps API.

    This class provides functionality to download SVIs at specific geographic coordinates,
    check image availability, and download images in a grid pattern across an area.

    Attributes:
        api_key (str): Google Maps API key for authentication.
        save_dir (str): Directory path to save downloaded images.
        base_url (str): Base URL for the Google Street View API.
        meta_url (str): URL for the metadata endpoint of the Google Street View API.
    """

    def __init__(self, api_key: str = None, save_dir: str = "images"):
        """
        Initialize the GoogleSVIDownloader with API key and save directory.

        If no API key is provided, it attempts to get the key from the GOOGLEMAP_API_KEY
        environment variable. If that fails, it prompts the user to enter the key.

        Args:
            api_key (str, optional): Google Maps API key. Defaults to None.
            save_dir (str, optional): Directory to save downloaded images. Defaults to "images".
        """
        if api_key is None:
            api_key = os.getenv("GOOGLEMAP_API_KEY")
            if api_key is None:
                api_key = getpass.getpass("Enter your Google Map API key: ")
        self.api_key = api_key
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        self.meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    def is_svi_available(self, lat: float, lon: float) -> bool:
        """
        Check if Street View imagery is available at the given coordinates.

        This method queries the Google Street View metadata API to determine
        whether street view imagery exists for the specified location.

        Args:
            lat (float): Latitude coordinate.
            lon (float): Longitude coordinate.

        Returns:
            bool: True if street view imagery is available, False otherwise.
        """
        params = {
            "location": f"{lat},{lon}",
            "key": self.api_key
        }
        response = requests.get(self.meta_url, params=params)
        data = response.json()
        return data.get("status") == "OK" and data.get("pano_id") is not None

    def download_svi(self, lat: float, lon: float, heading: int = 0,
                     pitch: int = 0, fov: int = 90, size: str = "640x640",
                     save: bool = True) -> Image.Image | None:
        """
        Download a Street View image at the specified coordinates with given parameters.

        Args:
            lat (float): Latitude coordinate.
            lon (float): Longitude coordinate.
            heading (int, optional): Camera heading in degrees, 0-360. Defaults to 0 (north).
            pitch (int, optional): Camera pitch in degrees, -90 to 90. Defaults to 0 (flat).
            fov (int, optional): Field of view in degrees, 0-120. Defaults to 90.
            size (str, optional): Image size in format "WIDTHxHEIGHT". Defaults to "640x640".
            save (bool, optional): Whether to save the image to disk. Defaults to True.

        Returns:
            PIL.Image.Image | None: Downloaded image as PIL Image object, or None if download failed.
        """
        params = {
            "location": f"{lat},{lon}",
            "size": size,
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": self.api_key
        }
        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            if save:
                filename = f"lat{lat:.6f}_lon{lon:.6f}_hdg{heading}.jpg"
                filepath = os.path.join(self.save_dir, filename)
                image.save(filepath)
                print(f"Saved SVI: {filepath}")
            return image
        else:
            print(f"Failed to download SVI at ({lat}, {lon}), status code: {response.status_code}")
            return None

    def generate_grid_coords(self, lat_start: float, lon_start: float,
                             rows: int, cols: int, delta: float) -> list[tuple[float, float]]:
        """
        Generate a grid of geographic coordinates starting from a point.

        Creates a rectangular grid of coordinates with the specified number of rows and columns,
        where each point is separated by the delta value in degrees.

        Args:
            lat_start (float): Starting latitude for the grid.
            lon_start (float): Starting longitude for the grid.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            delta (float): Spacing between points in degrees.

        Returns:
            list[tuple[float, float]]: List of (latitude, longitude) coordinate pairs.
        """
        coords = []
        for i in range(rows):
            for j in range(cols):
                lat = lat_start + i * delta
                lon = lon_start + j * delta
                coords.append((lat, lon))
        return coords

    def download_grid_svis(self, lat_start: float, lon_start: float, rows: int, cols: int,
                           delta: float, heading: int = 0, pitch: int = 0,
                           fov: int = 90, size: str = "640x640") -> list[dict]:
        """
        Download Street View images for a grid of locations.

        This method generates a grid of coordinates and downloads SVIs for each location
        where imagery is available.

        Args:
            lat_start (float): Starting latitude for the grid.
            lon_start (float): Starting longitude for the grid.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            delta (float): Spacing between grid points in degrees.
            heading (int, optional): Camera heading in degrees. Defaults to 0.
            pitch (int, optional): Camera pitch in degrees. Defaults to 0.
            fov (int, optional): Field of view in degrees. Defaults to 90.
            size (str, optional): Image size. Defaults to "640x640".

        Returns:
            list[dict]: List of dictionaries, each containing 'lat', 'lon', and 'image' keys
                        for the downloaded images.
        """
        coords = self.generate_grid_coords(lat_start, lon_start, rows, cols, delta)
        svis = []

        for lat, lon in coords:
            if self.is_svi_available(lat, lon):
                img = self.download_svi(lat, lon, heading, pitch, fov, size)
                if img:
                    svis.append({"lat": lat, "lon": lon, "image": img})
            else:
                print(f"No SVI available at ({lat}, {lon})")

        return svis

# === Segmentation and saving ===
def segment_and_save_images(
    image_pil: Image.Image,
    processor,
    model,
    colors = None,
    save_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None
) -> np.ndarray:
    """
    Perform semantic segmentation on an input image and optionally save the results.

    This function takes a PIL image, runs it through a semantic segmentation model,
    and creates a colored segmentation mask. The segmentation can be saved to disk
    with a specified filename prefix.

    Args:
        image_pil (PIL.Image.Image): Input image to segment.
        processor: HuggingFace image processor object for the segmentation model.
        model: HuggingFace segmentation model (e.g., SegFormer).
        colors (list, optional): List of RGB color tuples for visualization.
                                If None, uses CITYSCAPES_COLORS. Defaults to None.
        save_dir (str, optional): Directory to save segmentation results.
                                 If None, no saving occurs. Defaults to None.
        filename_prefix (str, optional): Prefix for the saved segmentation file.
                                        Defaults to None.

    Returns:
        np.ndarray: Segmentation mask array with shape (H, W) containing class indices.
    """
    if colors is None:
        colors = CITYSCAPES_COLORS

    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    # Resize mask to original image size
    mask_img = Image.fromarray(pred_mask.astype(np.uint8))
    mask_img = mask_img.resize(image_pil.size, resample=Image.NEAREST)
    mask_np = np.array(mask_img)

    # Create color mask for visualization
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for label_idx, color in enumerate(colors):
        color_mask[mask_np == label_idx] = color

    color_mask_img = Image.fromarray(color_mask)

    # Save segmentation image if directory and prefix provided
    if save_dir and filename_prefix:
        os.makedirs(os.path.join(save_dir, "segmentations"), exist_ok=True)
        color_mask_img.save(os.path.join(save_dir, "segmentations", f"{filename_prefix}_segmentation.png"))

    return mask_np

# === Visualization ===
def visualize_segmentation_pair(
    image_pil: Image.Image,
    mask_np: np.ndarray,
    show_legend: bool = True,
    classes=None,
    colors=None
) -> None:
    """
    Visualize an original image alongside its semantic segmentation mask.

    This function creates a side-by-side visualization of an input image and its
    corresponding segmentation mask, with an optional legend showing the semantic
    classes present in the segmentation.

    Args:
        image_pil (PIL.Image.Image): Original input image.
        mask_np (np.ndarray): Segmentation mask with class label indices as values.
        show_legend (bool, optional): Whether to display a legend with class names. Defaults to True.
        classes (list, optional): List of class names for the legend.
                                 If None, uses CITYSCAPES_CLASSES. Defaults to None.
        colors (list, optional): List of RGB color tuples for visualizing the mask.
                               If None, uses CITYSCAPES_COLORS. Defaults to None.

    Returns:
        None: This function displays the visualization but does not return any value.
    """
    if classes is None:
        classes = CITYSCAPES_CLASSES
    if colors is None:
        colors = CITYSCAPES_COLORS

    # Map label indices to RGB color mask using Cityscapes palette
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for label_idx, color in enumerate(colors):
        color_mask[mask_np == label_idx] = color

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_pil)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.title("Segmentation Prediction")
    plt.axis("off")

    if show_legend:
        unique_labels = np.unique(mask_np)
        handles = [
            mpatches.Patch(color=np.array(colors[idx]) / 255.0, label=classes[idx])
            for idx in unique_labels if idx < len(classes)
        ]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., title="Cityscapes Classes")

    plt.tight_layout()
    plt.show()

def batch_segment_and_visualize(
    save_dir: str,
    output_dir: str,
    processor,
    model,
    max_visualize: int = 6,
    image_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    """
    Process and segment a batch of images, save the results, and visualize a subset of them.

    This function processes all images in the specified directory, performs semantic segmentation
    on each one, saves the colored segmentation masks, and optionally visualizes the first few
    results as side-by-side comparisons.

    Args:
        save_dir (str): Directory containing the input images to process.
        output_dir (str): Directory where segmentation results will be saved.
        processor: HuggingFace image processor for preparing inputs to the segmentation model.
        model: HuggingFace segmentation model to use for inference.
        max_visualize (int, optional): Maximum number of images to visualize. Defaults to 6.
        image_extensions (tuple, optional): File extensions to identify images for processing.
                                          Defaults to (".jpg", ".png", ".jpeg").

    Returns:
        None: This function does not return a value but saves segmentation masks to disk
              and displays visualizations for the first max_visualize images.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_filenames = [f for f in os.listdir(save_dir) if f.lower().endswith(image_extensions)]

    for i, filename in enumerate(image_filenames):
        image_path = os.path.join(save_dir, filename)
        image_pil = Image.open(image_path).convert("RGB")

        # Run segmentation and save masks
        mask_np = segment_and_save_images(
            image_pil=image_pil,
            processor=processor,
            model=model,
            save_dir=output_dir,
            filename_prefix=os.path.splitext(filename)[0]
        )

        # Visualize some of the images
        if i < max_visualize:
            visualize_segmentation_pair(image_pil, mask_np, show_legend=True)
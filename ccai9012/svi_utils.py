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
    def __init__(self, api_key: str = None, save_dir: str = "images"):
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
    Perform segmentation on input PIL image, optionally save segmentation results.

    Args:
        image_pil (PIL.Image): Input image.
        processor: Huggingface processor for segmentation model.
        model: Huggingface segmentation model.
        save_dir (str, optional): Directory to save segmentation image. If None, no saving.
        filename_prefix (str, optional): Filename prefix for saved image.

    Returns:
        np.ndarray: Segmentation mask array (H, W) with label indices.
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
    Visualize original image and segmentation side-by-side.

    Args:
        image_pil (PIL.Image): Original input image.
        mask_np (np.ndarray): Segmentation mask with label indices.
        show_legend (bool): Whether to show legend of classes.
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
    Batch run segmentation on all images in save_dir, save results to output_dir,
    and visualize the first max_visualize images.

    Args:
        save_dir (str): Directory containing images to segment.
        output_dir (str): Directory to save segmentation masks.
        processor: Huggingface processor for segmentation.
        model: Huggingface segmentation model.
        max_visualize (int): Max number of images to visualize.
        image_extensions (tuple): Allowed image file extensions.
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
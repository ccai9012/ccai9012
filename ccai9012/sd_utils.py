"""
Stable Diffusion Utilities Module
=================================

This module provides utilities for text-to-image generation using Stable Diffusion models.
It offers a flexible interface to generate images either through the Hugging Face Inference API
(cloud-based) or locally using downloaded models.

The module is designed to simplify the process of generating images from text prompts,
handling authentication, model loading, and providing consistent interfaces regardless
of the execution mode chosen.

Main components:
- API key management: Secure handling of Hugging Face API keys
- SDClient: A versatile client that can operate in two modes:
  - "inference": Uses Hugging Face's Inference API (cloud-based, faster startup)
  - "local": Loads and runs models locally (higher throughput for multiple generations)

Usage:
    client = SDClient(mode="inference")  # Use Hugging Face Inference API
    images = client.generate_images("a photo of a cat", num_images=2)
"""

import os
import getpass
from huggingface_hub import InferenceClient
from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display


def get_hf_api_key(env_var: str = "HUGGINGFACE_API_KEY") -> str:
    """
    Get Hugging Face API key from environment variables or prompt user securely.

    This function first attempts to retrieve the API key from the specified environment variable.
    If the key is not found, it prompts the user to enter it securely (without displaying it),
    then stores it in the environment variable for future use within the session.

    Args:
        env_var (str): The name of the environment variable to check for the API key.
                      Defaults to "HUGGINGFACE_API_KEY".

    Returns:
        str: The Hugging Face API key.
    """
    api_key = os.getenv(env_var)
    if not api_key:
        api_key = getpass.getpass(f"Enter your {env_var}: ")
        os.environ[env_var] = api_key
    return api_key


class SDClient:
    """
    Stable Diffusion Client for text-to-image generation.

    This class provides a unified interface for generating images from text prompts,
    supporting two operational modes:

    1. "inference" mode: Uses Hugging Face's Inference API (cloud-based)
       - Advantages: Faster startup, no model downloads, lower memory requirements
       - Use case: Quick testing, limited hardware resources

    2. "local" mode: Loads and runs models locally via diffusers library
       - Advantages: Higher throughput for multiple generations, more control
       - Use case: Batch processing, offline usage, custom pipelines

    The class handles model loading, device management, and API authentication
    automatically, providing a simple interface for generating images.

    Attributes:
        mode (str): Operating mode ("inference" or "local").
        model_id (str): Hugging Face model repository ID.
        device (str): Torch device for computation ("cuda" or "cpu").
        api_key (str): Hugging Face API key for authentication.
        client: Inference client (in "inference" mode).
        pipe: StableDiffusionPipeline instance (in "local" mode).
        cache_dir (str): Directory to cache downloaded models (in "local" mode).
    """

    def __init__(
        self,
        mode: str = "inference",  # "inference" or "local"
        model_id: str = "stabilityai/stable-diffusion-2-base",
        cache_dir: str = None,
        use_auth_token: str = None,
        device: str = None,
    ):
        """
        Initialize the Stable Diffusion Client with specified configuration.

        Sets up the client based on the chosen operational mode, handling authentication,
        model loading, and device selection automatically.

        Args:
            mode (str): Operational mode for image generation. Options:
                       - "inference": Use Hugging Face's cloud-based Inference API.
                       - "local": Load and run the model locally using diffusers library.
                       Defaults to "inference".
            model_id (str): Hugging Face model repository ID for the Stable Diffusion model.
                          Defaults to "stabilityai/stable-diffusion-2-base".
            cache_dir (str, optional): Local directory to cache downloaded models in "local" mode.
                                     If None, defaults to "./models" as an absolute path.
            use_auth_token (str, optional): Hugging Face API token for authentication.
                                          If None, will attempt to retrieve or prompt for it.
            device (str, optional): PyTorch device to use for computation.
                                  If None, automatically selects CUDA if available, otherwise CPU.

        Raises:
            ValueError: If the specified mode is not "inference" or "local".
        """
        self.mode = mode.lower()
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.api_key = use_auth_token or get_hf_api_key()

        if self.mode == "inference":
            self.client = InferenceClient(
                provider="hf-inference",
                headers={"X-Use-Cache": "false"},
                api_key=self.api_key,
            )
        elif self.mode == "local":
            if cache_dir is None:
                cache_dir = os.path.abspath("./models")
            self.cache_dir = cache_dir
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                use_auth_token=self.api_key,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            ).to(self.device)
        else:
            raise ValueError("mode must be 'inference' or 'local'")

    def generate_images(
        self,
        prompt: str,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int = None,
        display_images: bool = True,
    ):
        """
        Generate images from a text prompt using Stable Diffusion.

        This method generates one or more images based on the provided text prompt,
        using either the Hugging Face Inference API (in "inference" mode) or a locally
        loaded model (in "local" mode). Generated images can optionally be displayed
        inline in Jupyter notebooks.

        Args:
            prompt (str): Text description of the image to generate.
            num_images (int, optional): Number of images to generate. Defaults to 1.
            guidance_scale (float, optional): Classifier-free guidance scale, controlling how
                                           closely the image follows the prompt. Higher values
                                           give more prompt adherence but less diversity.
                                           Defaults to 7.5.
            num_inference_steps (int, optional): Number of denoising steps in the diffusion process.
                                              More steps typically yield higher quality images
                                              but take longer to generate. Defaults to 50.
            seed (int, optional): Random seed for reproducible generation.
                                If None, a random seed is used. Defaults to None.
            display_images (bool, optional): Whether to display generated images inline
                                          in Jupyter notebook environments. Defaults to True.

        Returns:
            list: A list of PIL.Image objects representing the generated images.
        """
        images = []

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        if self.mode == "inference":
            for _ in range(num_images):
                img = self.client.text_to_image(
                    prompt=prompt,
                    model=self.model_id,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                )
                images.append(img)
        else:  # local mode
            for _ in range(num_images):
                img = self.pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
                images.append(img)

        if display_images:
            for img in images:
                display(img)

        return images

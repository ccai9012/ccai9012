# utils/sd_utils.py

import os
import getpass
from huggingface_hub import InferenceClient
from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display


def get_hf_api_key(env_var: str = "HUGGINGFACE_API_KEY") -> str:
    """Get HF API key from env or prompt user securely."""
    api_key = os.getenv(env_var)
    if not api_key:
        api_key = getpass.getpass(f"Enter your {env_var}: ")
        os.environ[env_var] = api_key
    return api_key


class SDClient:
    def __init__(
        self,
        mode: str = "inference",  # "inference" or "local"
        model_id: str = "stabilityai/stable-diffusion-2-base",
        cache_dir: str = None,
        use_auth_token: str = None,
        device: str = None,
    ):
        """
        Initialize SDClient.

        Args:
            mode: "inference" to use HuggingFace Inference API,
                  "local" to load model locally via diffusers.
            model_id: HuggingFace model repo ID.
            cache_dir: local directory to cache model (for local mode).
            use_auth_token: HF API token (str), if None will auto-get.
            device: torch device, default: "cuda" if available else "cpu".
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
        Generate images from prompt.

        Returns list of PIL.Image objects.

        For `inference` mode, uses HF Inference API.
        For `local` mode, uses local model pipeline.

        Args:
            prompt: Text prompt.
            num_images: Number of images to generate.
            guidance_scale: CFG scale.
            num_inference_steps: Diffusion steps.
            seed: Random seed.
            display_images: If True, show images inline (in notebooks).
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
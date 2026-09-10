"""
CCAI9012 Toolkit
===============

This package provides a collection of utilities for AI course projects, offering tools for
various machine learning, computer vision, and natural language processing tasks.

Modules:
    - llm_utils: Utilities for working with Large Language Models
    - nn_utils: Neural network training and evaluation utilities
    - sd_utils: Stable Diffusion image generation utilities
    - svi_utils: Google Street View Image handling utilities
    - viz_utils: Data and model visualization utilities
    - yolo_utils: YOLO object detection and tracking utilities
    - multi_modal_utils: Multi-modal AI model utilities
    - gan_utils: Generative Adversarial Network utilities

Each module contains specialized functions and classes to simplify common AI tasks,
from data preparation to model training, evaluation, and visualization.
"""

# Import stable repository paths before the optional utility modules.  These
# names are intentionally available from ``ccai9012`` for notebook use.
from .paths import (
    CACHE_DIR,
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PACKAGE_ROOT,
    REPOSITORY_ROOT,
    STARTER_KITS_DIR,
    WEEKLY_SCRIPTS_DIR,
    ensure_output_dir,
)

# Import all submodules
from . import llm_utils
from . import nn_utils
from . import sd_utils
from . import svi_utils
from . import viz_utils
from . import yolo_utils
from . import multi_modal_utils
from . import gan_utils

# Define the public API
__all__ = [
    "llm_utils",
    "nn_utils",
    "sd_utils",
    "svi_utils",
    "viz_utils",
    "yolo_utils",
    "multi_modal_utils",
    "gan_utils",
    "PACKAGE_ROOT",
    "REPOSITORY_ROOT",
    "STARTER_KITS_DIR",
    "WEEKLY_SCRIPTS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "CACHE_DIR",
    "OUTPUT_DIR",
    "ensure_output_dir",
]

# Define version
__version__ = "1.0.0"

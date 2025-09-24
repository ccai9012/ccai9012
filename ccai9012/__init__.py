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
]

# Define version
__version__ = "1.0.0"

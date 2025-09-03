# CCAI9012 Course Material and Toolkits

Artificial Intelligence is rapidly reshaping how we think, create, and solve problems across disciplines. From automation to creative expression, AI is transforming not only industries but also the way we engage with the world around us. This course explores AI as a tool for critical inquiry, design thinking, and communication—especially in the context of urban life and the built environment.

In this repository, you'll find a comprehensive set of resources, including weekly course materials and project starter kits, designed to help you harness the power of AI in your work.

## Repository Structure

```
├── weekly_scripts/        # Weekly course code
│   ├── week1/            # Python basics
│   ├── week2/            # Data processing & MLP
│   ├── week3/            # CNN & SVI
│   ├── week4/            # Image Generation & LLM
│   ├── week6/            # ML Fundamentals
│   ├── week7/            # YOLO Object Detection
│   ├── week8/            # Model Fairness
│   └── week9/            # ...
│
├── starter_kits/         # Project starter kits
│   ├── 1_traditional_generative_ml/    # Traditional generative models
│   ├── 2_llm_structure_output/         # LLM structured output
│   ├── 3_multimodal_reasoning/         # Multimodal reasoning
│   ├── 4_cv_models/                    # Computer vision models
│   └── 5_bias_detection_interpretability/  # Bias detection & interpretability
│
├── ccai9012/            # Core utility library
├── data/                # Example datasets
└── models/              # Pre-trained models
```

### weekly_scripts
Weekly course materials including:
- Tutorial code (`*_t_*.ipynb`)
- In-class exercises (`*_ic_*.ipynb`)
- Related datasets and resources

### starter_kits
Starter kits for final projects covering various AI applications:
1. Traditional Generative Models (e.g., GANs)
2. LLM Structured Output Processing
3. Multimodal Reasoning Applications
4. Computer Vision Model Applications
5. AI Bias Detection & Interpretability Analysis

Each kit contains complete example code and detailed documentation to serve as a reference and starting point for final projects.

## Installation Guide

### 1. Install Anaconda

First, install Anaconda, a Python data science platform that includes necessary tools and package management.

1. Visit [Anaconda's website](https://www.anaconda.com/products/distribution)
2. Download the installer for your operating system:
   - Windows: Download the .exe installer
   - macOS: Download the .pkg installer
   - Linux: Download the .sh installer
3. Run the installer with default options

Verify installation:
1\. Open Terminal:  
   - On **Windows**: Press `Win + S`, type `cmd` , and press Enter.  
   - On **macOS**: Press `Command + Space`, type `Terminal`, and press Enter.

2\. Copy and paste the following command into the terminal, then press Enter:
```bash
conda --version
```

### 2. Create Virtual Environment

Create a new environment named `ccai9012` with Python 3.9：

```bash
conda create -n ccai9012 python=3.9
```

Activate the environment：
```bash
conda activate ccai9012
```

### 3. Install Dependencies

Clone the project and install dependencies：

```bash
# Clone repository
git clone [repository_url]
cd toolkit

# Install requirements
pip install -r requirements.txt

# Install toolkit in development mode
pip install -e .
```

### 4. Verify Installation

Verify in Python：
```python
import ccai9012
import torch
import transformers
```

If no errors occur, the installation is successful.

## Common Issues

### CUDA Setup
If you have an NVIDIA GPU, it's recommended to set up CUDA for GPU acceleration：

1. Check CUDA compatibility：
```bash
nvidia-smi
```

2. Install matching CUDA version：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Support & Feedback

For questions or suggestions：
1. Submit an issue
2. Email [course_email]
3. Post in the course forum

## License

[Add license information]

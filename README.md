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
│   ├── 1_traditional_generative_ml/
│   ├── 2_llm_structure_output/
│   ├── 3_multimodal_reasoning/
│   ├── 4_cv_models/
│   └── 5_bias_detection_interpretability/
│
├── ccai9012/            # Core utility library
├── docs/                # Documentation
│   └── ccai9012/        # Unified documentation website
├── data/                # Example datasets
└── models/              # Pre-trained models
```

## 📚 Documentation

**[View Complete Documentation Website →](docs/ccai9012/home.html)**

The complete course documentation is now available as a unified website with the following sections:

- **[Home](docs/ccai9012/home.html)** - Course overview and quick start guide
- **[Installation Guide](docs/ccai9012/installation.html)** - Step-by-step setup instructions
- **[Starter Kits](docs/ccai9012/starter_kits.html)** - Five project modules with examples and use cases
- **[Reading Materials](docs/ccai9012/reading_material.html)** - Curated academic papers and resources
- **[Datasets Reference](docs/ccai9012/datasets.html)** - Comprehensive catalog of 40+ datasets
- **[API Documentation](docs/ccai9012/index.html)** - Complete ccai9012 library reference

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ccai9012/ccai9012.git
cd ccai9012
```

### 2. Set Up Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ccai9012

# Install the package
pip install -e .
```

### 3. Launch Jupyter Notebook

```bash
# Add kernel to Jupyter
python -m ipykernel install --user --name ccai9012 --display-name "ccai9012"

# Start Jupyter
jupyter notebook
```

### 4. Browse Documentation

Open `docs/ccai9012/home.html` in your web browser to access the complete documentation website.

## Starter Kit Modules

1. **Traditional Generative Models** - GANs for architectural pattern generation
2. **LLM Structured Output** - Text analysis and knowledge extraction
3. **Multimodal Reasoning** - Visual-language AI applications
4. **Computer Vision Models** - Detection, tracking, and segmentation
5. **Bias Detection & Interpretability** - Ethical AI and model transparency

## Support & Feedback

For questions or suggestions:
1. Submit an issue on our GitHub repository
2. Email [course_email]
3. Post in the course forum

## License

[Add license information]

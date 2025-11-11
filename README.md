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
├── docs/                # Documentation
│   ├── Reading_Material.md     # Course reading materials
│   ├── starter_kits.md         # Detailed starter kit documentation
│   └── ccai9012/              # Module API documentation
├── data/                # Example datasets
└── models/              # Pre-trained models
```

### weekly_scripts
Weekly course materials including:
- Tutorial code (`*_t_*.ipynb`)
- In-class exercises (`*_ic_*.ipynb`)
- Related datasets and resources

### starter_kits
Starter kits for final projects covering various AI applications.

**[View detailed starter kit documentation](docs/starter_kits.md)**

Quick overview:

1. **Traditional Generative Models**: Create AI-generated images and patterns using GANs (Generative Adversarial Networks). 
   - `GANmapper` project demonstrates how to generate architectural or urban patterns from existing imagery.

2. **LLM Structured Output Processing**: Extract and analyze structured information from text using Large Language Models. Includes projects for:
   - `airbnb_reviews`: Analyzing sentiment and feature preferences from accommodation reviews
   - `lit_review`: Automating literature review summaries from academic papers
   - `pdf_extraction`: Extracting structured information from PDF documents
   - `urban_sentiment`: Analyzing public sentiment about urban spaces from text data

3. **Multimodal Reasoning Applications**: Combine image and text understanding for advanced AI applications:
   - `clip_historic`: Classify architectural styles and historic buildings using CLIP models
   - `gen_images_eval`: Generate and evaluate AI-created images based on text prompts

4. **Computer Vision Model Applications**: Apply computer vision to urban analysis:
   - `svi_housing_price`: Predict housing prices from Street View imagery
   - `webcam_yolo`: Real-time object detection for pedestrian and traffic analysis

5. **AI Bias Detection & Interpretability**: Explore ethical AI and model transparency:
   - `credit_audit`: Audit credit decision models for potential biases


### docs
Comprehensive documentation for the course:
- **[Reading Materials](docs/reading_material.md)**: Curated articles, papers, and resources organized by learning modules
- **[📖 Reading Materials](docs/reading_material.md)**: Curated articles, papers, and resources organized by learning modules
- **[🚀 Starter Kits Guide](docs/starter_kits.md)**: Detailed guides for each project starter kit with use cases, datasets, and required packages
- **[📊 Datasets Reference](docs/datasets.md)**: Comprehensive dataset catalog with 40+ datasets and direct download links
- **[📚 API Documentation](docs/ccai9012/index.html)**: HTML documentation for the `ccai9012` library with usage examples and parameter descriptions
**Access API documentation by opening `/docs/ccai9012/index.html` in your web browser**

## Installation Guide

This guide will help you set up the toolkit on your computer, even if you're new to programming or AI.

### 1. Install Anaconda

First, install Anaconda, a Python data science platform that includes necessary tools and package management.

1. Visit [Anaconda's website](https://www.anaconda.com/products/distribution)
2. Download the installer for your operating system:
   - **Windows**: Download the .exe installer (Select "64-Bit Graphical Installer")
   - **macOS**: Download the .pkg installer (Select "64-Bit Graphical Installer")
   - **Linux**: Download the .sh installer

3. Run the installer:
   - **Windows**: Double-click the .exe file and follow the prompts. Important: Select "Add Anaconda to my PATH environment variable" during installation.
   - **macOS**: Double-click the .pkg file and follow the installation wizard.
   - **Linux**: Open a terminal, navigate to the download location, and run: `bash Anaconda3-xxxx.xx-Linux-x86_64.sh` (replace xxxx.xx with the version you downloaded)

Verify installation:
1\. Open Terminal:  
   - On **Windows**: Press `Win + S`, type `Anaconda Prompt`, and select "Anaconda Prompt (anaconda3)".  
   - On **macOS**: Press `Command + Space`, type `Terminal`, and press Enter.
   - On **Linux**: Open your terminal application.

2\. Copy and paste the following command into the terminal, then press Enter:
```bash
conda --version
```
You should see the conda version number displayed, confirming installation.

### 2. Download the Project and Create Environment

Clone the project repository or download it as a ZIP file, then create the conda environment:

**Option 1: Using Git (Recommended)**
```bash
# Navigate to your desired directory
cd ~/Desktop  # Or any directory where you want to store the project

# Clone repository (you need Git installed)
git clone https://github.com/ccai9012/ccai9012.git
cd ccai9012

# Create environment from environment.yml
conda env create -f environment.yml
```

**Option 2: Download ZIP**
If you don't have Git installed:
- Visit the GitHub repository page
- Click the green "Code" button and select "Download ZIP"
- Extract the ZIP file to your desired location
- Open terminal and navigate to the extracted folder:
```bash
cd path/to/extracted/ccai9012 # use your own path, for example: cd ~/Desktop/ccai9012
# Create environment from environment.yml
conda env create -f environment.yml
```

This command will:
- Create a new conda environment named `ccai9012`
- Install Python and all required packages
- Set up all dependencies for the course materials

Activate the environment:
```bash
conda activate ccai9012 # ensuring you're in the project directory, with (ccai9012) yourname@device toolkit % displayed in terminal
```
Your command prompt should now show `(ccai9012)` at the beginning of the line, indicating the environment is active.

Install the ccai9012 package in development mode:
```bash
# Make sure you're in the project directory and environment is activated
pip install -e .
```
This installs the ccai9012 utilities as a package, allowing you to `import ccai9012` from anywhere.

### 3. Set Up Jupyter Notebook Kernel

To use the course materials in Jupyter notebooks, you need to add the conda environment as a Jupyter kernel:

```bash
# Make sure the ccai9012 environment is activated
conda activate ccai9012

# Install the environment as a Jupyter kernel
python -m ipykernel install --user --name ccai9012 --display-name "ccai9012"
```

### 4. Launch Jupyter Notebook

Now you can start Jupyter Notebook and access all course materials:

```bash
# Make sure you're in the project directory and environment is activated
conda activate ccai9012 # activate the environment if not already done
cd path/to/ccai9012  # Navigate to the project directory if not already there, use your own path

# Launch Jupyter Notebook
jupyter notebook
```

**Important Jupyter Setup Steps:**

1. **Jupyter will open in your web browser** (usually at http://localhost:8888)
2. **Select the correct kernel**: When you open any notebook (.ipynb file):
   - Click on "Kernel" in the menu bar
   - Select "Change kernel"
   - Choose "ccai9012" from the dropdown
   - This ensures the notebook uses the correct environment with all installed packages

3. **Navigate to course materials**:
   - `weekly_scripts/` - for weekly course materials
   - `starter_kits/` - for project starter kits
   - `docs/ccai9012/` - for documentation (open index.html in browser)

## Common Issues

### CUDA Setup
If you have an NVIDIA GPU, it's recommended to set up CUDA for GPU acceleration：

1. Check CUDA compatibility：
```bash
nvidia-smi
```
This will display your GPU and driver information.

2. Install matching CUDA version：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Common Error Messages

1. **ModuleNotFoundError**: If you see "No module named 'xyz'", run: `pip install xyz`
2. **DLL Load Failed**: On Windows, reinstall PyTorch with the correct CUDA version
3. **Out of Memory**: Reduce batch sizes in your code or use CPU mode

## Support & Feedback

For questions or suggestions：
1. Submit an issue on our GitHub repository
2. Email [course_email]
3. Post in the course forum

## License

[Add license information]

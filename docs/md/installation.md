# Installation Guide

This guide will help you set up the toolkit on your computer, even if you're new to programming or AI.

## Before You Start

Allow about 30–60 minutes for the first setup. You need an internet connection, permission to install software, and approximately **2.5 GB of free disk space** in total. Keep this page open while you work, and run one command at a time.

You will use four pieces that work together:

- **Python** is the programming language used by the course materials.
- **Conda** installs Python and the course dependencies inside an isolated environment, so they do not interfere with other projects on your computer.
- **Jupyter Notebook** is the browser-based interface in which you open and run the course notebooks.
- **The `ccai9012` package** contains reusable course functions that the notebooks import.

The setup sequence is therefore: install Anaconda → download this repository → create and activate the `ccai9012` Conda environment → install the course package → launch Jupyter with the `ccai9012` kernel.

### Terminal Basics: Open the Course Folder

The installation commands must be run inside the downloaded `ccai9012` folder. The only folder command you need is `cd`, which means **change directory**. It tells the terminal to move into another folder.

The path is written differently on macOS and Windows. In the examples below, `xxx` represents a work directory that you choose. Replace it with the real name of your folder.

| What you want to do | macOS Terminal | Windows Anaconda Prompt |
| --- | --- | --- |
| Open the command application | Press `Command + Space`, then open **Terminal** | Press `Win + S`, then open **Anaconda Prompt** |
| Move to your work directory before downloading the repository | `cd ~/xxx` | `cd "%USERPROFILE%\xxx"` |

## 1. Install Anaconda

First, install Anaconda, a Python data science platform that includes necessary tools and package management.

### Download Anaconda

1. Visit [Anaconda's website](https://www.anaconda.com/products/distribution)
2. Download the installer for your operating system:
   - **Windows**: Download the .exe installer (Select "64-Bit Graphical Installer")
   - **macOS**: Download the .pkg installer (Select "64-Bit Graphical Installer")
   - **Linux**: Download the .sh installer

<p align="center">
  <img src="figs/install/SCR-20251218-moef.png" width="600"><br>
  <em>Choose the installer that matches your operating system and processor.</em>
</p>

### Run the Installer

- **Windows**: Double-click the .exe file and follow the prompts. **Important**: Select "Add Anaconda to my PATH environment variable" during installation.
- **macOS**: Double-click the .pkg file and follow the installation wizard.
- **Linux**: Open a terminal, navigate to the download location, and run:
  ```bash
  bash Anaconda3-xxxx.xx-Linux-x86_64.sh # replace xxxx.xx with the version you downloaded
  ```

### Verify Installation

**1. Open Terminal:**

- On **Windows**: Press `Win + S`, type `Anaconda Prompt`, and select "Anaconda Prompt (anaconda3)".
- On **macOS**: Press `Command + Space`, type `Terminal`, and press Enter.
- On **Linux**: Open your terminal application.

**2. Copy and paste the following command into the terminal, then press Enter:**

```bash
conda --version
```

You should see the conda version number displayed, confirming installation.

<p align="center">
  <img src="figs/install/SCR-20251218-mpki.png" width="600"><br>
  <em>Anaconda installation confirmation.</em>
</p>

## 2. Download the Project and Create Environment

Clone the project repository or download it as a ZIP file, then create the conda environment:

### Option 1: Using Git (Recommended)

First use the table above to move to the work directory where you want to keep the repository. Then run these commands on either macOS or Windows:

```bash
# Clone repository (you need Git installed)
git clone https://github.com/ccai9012/ccai9012.git
cd ccai9012
```

After `cd ccai9012`, the path shown before the cursor should end in `ccai9012`. Do not worry if the rest of the path looks different.

Now create the environment:

```bash
conda env create -f environment.yml
```

<div class="info-box">
If this command says that <code>environment.yml</code> cannot be found, the terminal is probably outside the repository. Run <code>cd ccai9012</code>, then try the command again.
</div>

<p align="center">
  <img src="figs/install/SCR-20251218-mxmc.png" width="600"><br>
  <em>A successful environment creation ends without an error and identifies the environment as “ccai9012”.</em>
</p>

### Option 2: Download ZIP

If you don't have Git installed:

- Visit the GitHub repository page
- Click the green "Code" button and select "Download ZIP"
- Extract the ZIP file to your desired location
- Open Terminal or Anaconda Prompt and use the second row of the table above to enter the work directory containing the extracted folder.
- Enter the extracted folder. Replace `ccai9012` if its actual folder name is different:

```bash
cd ccai9012
```

After entering the extracted folder, the path shown before the cursor should end in that folder's name. Then run:

```bash
conda env create -f environment.yml
```

<div class="info-box">
If this command says that <code>environment.yml</code> cannot be found, return to the extracted repository folder and try the command again.
</div>

This command will:

- Create a new conda environment named `ccai9012`
- Install Python and all required packages
- Set up all dependencies for the course materials

### Activate the Environment

```bash
conda activate ccai9012 # ensuring you're in the project directory
```

Your command prompt should now show `(ccai9012)` at the beginning of the line, indicating the environment is active.

<p align="center">
  <img src="figs/install/SCR-20251218-myst.png" width="600"><br>
  <em>The prompt shows the active environment as “(ccai9012)”; the highlighted path is the project directory.</em>
</p>

### Install the ccai9012 Package

Install the ccai9012 package in development mode:

```bash
# Make sure you're in the project directory and environment is activated
pip install -e .
```

This installs the ccai9012 utilities as a package, allowing you to `import ccai9012` from anywhere.

<p align="center">
  <img src="figs/install/SCR-20251218-mzpj.png" width="600"><br>
  <em>A successful editable install ends with a “Successfully installed” message.</em>
</p>

## 3. Test Your Environment

After installation, it's important to verify that all required packages are properly installed. Run the provided test script:

```bash
# Make sure the ccai9012 environment is activated
conda activate ccai9012

# Run the test script
python test_environment.py
```

### Understanding Test Results

The test script will check multiple categories of packages:

- **Core Scientific Packages**: numpy, pandas, matplotlib, scipy, scikit-learn
- **Deep Learning Frameworks**: PyTorch, torchvision, transformers, diffusers, accelerate
- **LLM Packages**: langchain, openai, tiktoken, and related tools
- **Computer Vision Tools**: OpenCV, YOLO (ultralytics), and related packages
- **Visualization Libraries**: plotly, seaborn, and plotting tools
- **Custom Utilities**: ccai9012 package modules

**Output Symbols:**
- ✓ (checkmark) = Package imported successfully
- ⚠ (warning) = Package works but may have version compatibility warnings
- ✗ (cross) = Package failed to import (needs troubleshooting)

<p align="center">
  <img src="figs/install/SCR-20251218-naog.png" width="600"><br>
  <em>Checkmarks confirm that the corresponding package imported successfully.</em>
</p>

### Verbose Testing

For detailed information about each test:

```bash
python test_environment.py --verbose
```

This will show each package being tested in real-time and provide more detailed error messages if any issues occur.

### What to Do If Tests Fail

If you see ✗ marks for any packages:

1. **Check the error message** - It usually indicates what's missing
2. **Reinstall the environment**:
   ```bash
   conda deactivate
   conda env remove -n ccai9012
   conda env create -f environment.yml
   conda activate ccai9012
   pip install -e .
   ```
3. **Install missing packages individually**:
   ```bash
   pip install package-name
   ```
4. **Check the Common Issues section** below for platform-specific problems

## 4. Set Up Jupyter Notebook Kernel

To use the course materials in Jupyter notebooks, you need to add the conda environment as a Jupyter kernel:

```bash
# Make sure the ccai9012 environment is activated
conda activate ccai9012

# Install the environment as a Jupyter kernel
python -m ipykernel install --user --name ccai9012 --display-name "ccai9012"
```

## 5. Launch Jupyter Notebook

Now you can start Jupyter Notebook and access all course materials:

If your terminal is not already inside the repository, use the appropriate command in the table above first. Then run:

```bash
conda activate ccai9012 # activate the environment if not already done
jupyter notebook
```

### Important Jupyter Setup Steps

1. **Jupyter will open in your web browser** (usually at http://localhost:8888)
2. **Select the correct kernel**: When you open any notebook (.ipynb file):
   - Click on "Kernel" in the menu bar
   - Select "Change kernel"
   - Choose "ccai9012" from the dropdown
   - This ensures the notebook uses the correct environment with all installed packages
3. **Navigate to course materials**:
   - `weekly_scripts/` - for weekly course materials
   - `starter_kits/` - for project starter kits
   - `docs/` - for the generated course website and API documentation

## Common Issues

### CUDA Setup

If you have an NVIDIA GPU, it's recommended to set up CUDA for GPU acceleration:

#### 1. Check CUDA compatibility:

```bash
nvidia-smi
```

This will display your GPU and driver information.

#### 2. Install matching CUDA version:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Common Error Messages

1. **ModuleNotFoundError**: If you see "No module named 'xyz'", run: `pip install xyz`
2. **DLL Load Failed**: On Windows, reinstall PyTorch with the correct CUDA version
3. **Out of Memory**: Reduce batch sizes in your code or use CPU mode

## Support & Feedback

For questions or suggestions:

1. Submit an issue on our GitHub repository
2. Email [course_email]
3. Post in the course forum

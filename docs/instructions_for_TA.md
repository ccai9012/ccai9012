# TA / Maintainer Handover Instructions

> Target audience: future TAs / maintainers (TA / course staff).
>
> This document answers:
> 1) How to install and verify the environment from scratch;
> 2) Repository structure and “where to find what”;
> 3) API token setup and usage;;
> 4) How to update the documentation system.
> 5) Other checklist.

The public collaboration workflow and maintainer-only authoring details now
live in separate Course Docs pages:

- `docs/collaboration_workflow.md`
- `docs/maintainer_authoring.md` (local only; not part of the public site)

Use the collaboration page for clone/edit/inspect/commit/pull-before-push,
conflict escalation, and public verification. Use the maintainer page for the
repository map, worked Markdown-page example, navigation registration, asset
diagnosis, and responsive checks. This local file remains a handover index.

---

## 0. Quick Navigation (the 5 entry points you’ll use most)

1. **Environment setup (for students/TAs)**: `docs/installation.md` (the same content is also converted to `docs/installation.html`)
2. **Repository overview**: root `README.md`
3. **Weekly exercises / in-class code**: `weekly_scripts/wip/`
4. **Final project Starter Kits**: `starter_kits/` (overview doc: `docs/md/starter_kits/index.md`)
5. **Toolkit API**: `ccai9012/` source; corresponding HTML: `docs/api/*.html`

---

## 1. Environment Setup (macOS / Windows / Linux)

This repository recommends using a Conda environment (default name: `ccai9012`). The repo provides both `environment.yml` and `requirements.txt`; prefer `environment.yml`.

### 1.1 Prerequisite: Install Anaconda / Miniconda

- Student-friendly: Anaconda (larger but fewer pitfalls)
- TA/developer: Miniconda is also fine

Verification:
- Run `conda --version` in a terminal; it should print a version.

### 1.2 Create the environment (main flow)

From the repository root (where `environment.yml` is located), run:

1. Create environment: `conda env create -f environment.yml`
2. Activate environment: `conda activate ccai9012`
3. Install this repo as an editable package: `pip install -e .`

Why `pip install -e .` is needed:
- This allows notebooks/scripts to `import ccai9012` directly without relying on the working directory.

### 1.3 Add a Jupyter Kernel

With the environment activated, run:

- `python -m ipykernel install --user --name ccai9012 --display-name "ccai9012"`

Then, when opening notebooks, select the `ccai9012` kernel.

### 1.4 Verify the environment (strongly recommended after every release)

The repository provides a one-command self-check:

- `python test_environment.py`

It attempts to import core dependencies (scientific stack, CV, LLM, diffusers, ultralytics, etc.) and reports missing/failed items.

### 1.5 Common pitfalls (quick reference for TA support)

- **pip/conda conflicts**: Prefer rebuilding from `environment.yml`; do not install packages into the base environment.
- **Model/data size**: `models/` and some datasets can be large; avoid committing them to git. In course instructions, tell students to “download on demand”.
- **GPU/CUDA**: This repo should run on CPU by default. If students have an NVIDIA GPU, install a matching PyTorch/CUDA version as needed (see the CUDA notes in `docs/installation.md`).

---

## 2. Repository Structure (“where to find what”)

Below are the directories TAs most commonly maintain:

### 2.1 `weekly_scripts/` (weekly course code)

- Organized by teaching schedule: one folder per week (e.g., `week1/`, `week2/`, …)
- Typically includes:
  - Lecture notebooks/scripts
  - Weekly demos, exercises, runnable examples
  - Possibly small datasets/assets needed for that week (if any)

The root README provides a week-by-week theme overview (e.g., week1 Python basics, week6 CNN/SVI, week10 YOLO, etc.).

**TA maintenance tips**:
- Aim for each week’s materials to be “independently runnable” (minimize dependencies on files from other weeks).
- Consider clear naming conventions: `lecture_*.ipynb`, `demo_*.ipynb`, `exercise_*.ipynb` (if you want to gradually standardize).

### 2.2 `starter_kits/` (final project templates / modular toolbox)

- Organized by project direction: grouped by topic modules rather than weeks.
- There are currently 5 modules (see `docs/starter_kits.md` for full explanations and example figures):
  1. `1_traditional_generative_ml/`: Traditional generative ML / GAN (see `ccai9012.gan_utils`)
  2. `2_llm_structure_output/`: LLM structured output (see `ccai9012.llm_utils`, `ccai9012.viz_utils`)
  3. `3_multimodal_reasoning/`: Multimodal reasoning (see `ccai9012.multi_modal_utils`, `ccai9012.svi_utils`)
  4. `4_cv_models/`: Detection/segmentation/tracking (see `ccai9012.yolo_utils`, etc.)
  5. `5_bias_detection_interpretability/`: Bias detection and interpretability (SHAP/AIF360, etc.)

**TA maintenance tips**:
- For each starter kit, try to provide:
  - A minimal runnable pipeline
  - A clear data placement convention (e.g., a `data/` subdirectory)
  - Comments on key parameters (e.g., API keys, model names, input formats)

### 2.3 `ccai9012/` (course toolkit source)

- Core reusable code for weekly scripts / starter kits.
- Files are roughly organized by function:
  - `llm_utils.py`: LLM calls, structured output, embeddings, etc.
  - `multi_modal_utils.py`: Multimodal utilities
  - `yolo_utils.py`: YOLO detection and other CV tools
  - `sd_utils.py`: Stable Diffusion / image generation
  - `svi_utils.py`: Street View Imagery utilities
  - `viz_utils.py`: Visualization helpers
  - `nn_utils.py`: Neural network utilities

After updating code here, TAs should:
- Run `python test_environment.py`
- Spot-check 1–2 weekly notebooks (or at least do an import + key-function smoke test)

### 2.4 `docs/` (documentation & static site generation)

- `docs/md/*.md` are source files; `.html` are generated pages; check `docs/README.md` for more details.

---

## 3. API Token Setup (for LLM / CV models / SVI)

This toolkit uses several external APIs. In most cases you’ll either distribute a course token to students, or ask students to use their own token.

Main platforms:

- **Hugging Face**
  - used for *most non-LLM models* and some classic dataset/model interfaces; 
  - online inference may charge by token/use with a monthly free tier; see the website for current policy
  - Token page: https://huggingface.co/settings/tokens
- **DeepSeek** (LLM-related) 
  - charges a small amount based on token usage):
  - Usage page: https://platform.deepseek.com/usage
- **Google Maps Platform** (SVI-related) 
  - usually requires adding billing / bank card info, so still need to be added
  - Platform home: https://mapsplatform.google.com/

How tokens are loaded/used (important):

- The token loading logic is centralized in `ccai9012/token_utils.py` (`get_token(...)`). Precedence is:
  1) `ccai9012/token.yaml` (or the file pointed to by `CCAI9012_TOKEN_YAML`)
  2) Environment variables
  3) If still missing, prompt for manual input when the API is used
- Students should put tokens into `ccai9012/token.yaml` under `env:` (e.g., `DEEPSEEK_API_KEY`, `HUGGINGFACE_API_KEY`, `GOOGLEMAP_API_KEY`).
- If a token isn’t set in the YAML (or env vars), they’ll be asked to input it at runtime when they hit a feature that needs it.

Platform account details and TA-level API tokens will be distributed to TAs separately.

---

## 4. Documentation Update Workflow (how to publish HTML after editing Markdown)

- **If you only changed documentation content (Markdown)**:
  - Edit the `.md` files under `docs/md/`
  - Then run `docs/md_to_html.py` to regenerate the site HTML

- **If you added/removed pages, or adjusted the sidebar/navigation structure**:
  - Add/move Markdown files under `docs/md/`
  - Update the navigation mapping in `docs/pages.json` (defines `source` → `output`, `children`, etc.)
  - Then run `docs/md_to_html.py`

- **If you changed the toolkit API (`ccai9012/` code) and want the API docs updated**:
  - First run `docs/generate_api_doc.py` to regenerate `docs/api/`
  - Then run `docs/md_to_html.py` to apply site styling/navigation integration

For more detailed directory conventions, execution order, and notes, refer to: `docs/README.md`.

---

## 5. Data and Models (most common in TA Q&A)

### 5.1 `data/`

- Stores course example data (e.g., `housing_price_california.csv`, `compas.db`, `german_credits/`, `yelp_reviews/`).
- General principles:
  - Small datasets can be shipped with the repo
  - Very large datasets should be provided via download links + a documented placement path

### 5.2 `models/`

- Used to cache/store pretrained models (often in a HuggingFace-cache-style directory layout).
- Can be very large; to ensure stable in-class demos, TAs may want to warm up by downloading ahead of time on TA machines.

---

## 6. Suggested checklist for new TAs 

### 6.1 First-time setup (30 minutes)

- [ ] Clone the repo
  - [ ] `conda env create -f environment.yml`
  - [ ] `conda activate ccai9012`
  - [ ] `pip install -e .`
  - [ ] `python test_environment.py`
- [ ] Open 1 notebook (e.g., week1) to confirm it runs
- [ ] API token setup: fill `ccai9012/token.yaml` (see Section 3) 

### 6.2 After any changes to environment/dependencies/core toolkit

- [ ] Rebuild the environment (or at least validate once in a fresh environment)
- [ ] Run `test_environment.py`
- [ ] Spot-check:
  - 1 LLM-related item
  - 1 CV/YOLO-related item
  - 1 diffusion / multimodal item (if used in the current offering)
- [ ] Rebuild documentation website HTML if docs changed: 
  - [ ] `python docs/generate_api_doc.py` (if API changed)
  - [ ] `python docs/md_to_html.py`

---

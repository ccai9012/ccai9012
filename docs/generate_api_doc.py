#!/usr/bin/env python3
"""Build the standalone Sphinx API site for :mod:`ccai9012`.

Module pages are generated from the package's syntax tree so only public objects
defined by each module enter the API. Sphinx autosummary then creates the
individual function, class, and method pages.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
SOURCE_DIR = DOCS_DIR / "api_source"
PACKAGE_DIR = REPO_ROOT / "ccai9012"
OUTPUT_DIR = DOCS_DIR / "api"

MODULE_METADATA = {
    "gan_utils": {
        "overview": (
            "This module supports a complete paired image-to-image translation exercise using a compact "
            "Pix2Pix-style GAN. It connects the steps that students often see separately: organizing aligned "
            "images, loading them as tensors, defining the generator and discriminator, training, and inference."
        ),
        "helps": (
            "turn region-based Source/Target folders into reproducible train and test splits",
            "apply synchronized augmentation so an input image stays aligned with its target",
            "train a teaching-scale U-Net generator and PatchGAN discriminator",
            "load checkpoints and generate output images for a test folder",
        ),
        "uses": (
            "You have paired images showing the same place or object in two domains, such as map-to-aerial, facade-to-segmentation, or sketch-to-image.",
            "You want to study the mechanics of adversarial and reconstruction losses before adopting a larger GAN framework.",
            "You need a transparent baseline for a course project; it is not intended as a production-scale GAN library.",
        ),
        "workflow": (
            "Prepare aligned image pairs with ``prepare_gan_dataset``.",
            "Create batches with ``create_paired_data_loader``.",
            "Initialize ``UNetGenerator`` and ``PatchDiscriminator``.",
            "Train with ``train_GAN``, then use ``load_model`` and ``inference_gan`` for prediction.",
        ),
        "starter_kits": (("Generative AI: GANs", "../../starter_kits/m1_gan.html"),),
    },
    "llm_utils": {
        "overview": (
            "This module provides the reusable building blocks behind the course's language-model examples. "
            "It covers direct prompting, repeated generation, review analysis, and retrieval-augmented question "
            "answering over PDF documents."
        ),
        "helps": (
            "initialize the supported chat model without placing credentials in a notebook",
            "send prompts and collect one or several model responses",
            "turn PDFs into chunks and a searchable vector retriever",
            "parse structured Markdown tables and analyze review datasets",
        ),
        "uses": (
            "You are building a text-generation or structured-output exercise.",
            "A question should be answered from a supplied PDF rather than from the model's general knowledge.",
            "You want course-ready helpers while keeping prompt design and result evaluation visible in the notebook.",
        ),
        "workflow": (
            "Load the credential and initialize the model.",
            "Prepare a prompt or build a PDF retriever, depending on the task.",
            "Invoke the model or QA chain and inspect the raw response.",
            "Parse, compare, or save the result for later evaluation.",
        ),
        "starter_kits": (("Large Language Models", "../../starter_kits/m2_llm.html"),),
    },
    "multi_modal_utils": {
        "overview": (
            "This module introduces models that connect images and language. ``CLIPClassifier`` compares an "
            "image with candidate text labels, while ``VisionQAProcessor`` asks open-ended questions about image "
            "content and can extract keywords from the response."
        ),
        "helps": (
            "perform zero-shot image classification with student-defined text categories",
            "batch-process an image folder and save comparable confidence scores",
            "generate captions or answers to questions about images",
            "extract recurring visual attributes for downstream analysis",
        ),
        "uses": (
            "Your categories are semantic descriptions rather than labels from a trained classifier.",
            "Your research question requires both visual evidence and natural-language interpretation.",
            "You need an exploratory multimodal baseline and will manually validate model responses.",
        ),
        "workflow": (
            "Choose CLIP classification or visual question answering based on the research question.",
            "Initialize the corresponding class and point it to an image collection.",
            "Run a single-image example before starting batch processing.",
            "Review confidence scores or generated text before aggregating the results.",
        ),
        "starter_kits": (("Multimodal AI", "../../starter_kits/m3_mm.html"),),
    },
    "nn_utils": {
        "overview": (
            "This module gathers the repeated data preparation, training, and evaluation steps used in the "
            "course's introductory neural-network examples. It keeps the training loop available for study while "
            "reducing notebook boilerplate."
        ),
        "helps": (
            "split and standardize tabular features before creating PyTorch data loaders",
            "select an available CPU, CUDA, or Apple Silicon device",
            "train a supplied model while recording loss history",
            "evaluate regression and classification predictions with familiar metrics",
        ),
        "uses": (
            "You are training a small supervised model on tabular or already-vectorized data.",
            "Several notebooks need the same split, loader, and evaluation conventions.",
            "You want to compare model behavior rather than rewrite infrastructure in each exercise.",
        ),
        "workflow": (
            "Separate features and targets, then call ``prepare_dataloaders``.",
            "Define a PyTorch model and choose a device with ``get_best_device``.",
            "Train the model with ``train_model``.",
            "Use the matching regression or classification evaluator to interpret performance.",
        ),
        "starter_kits": (),
    },
    "sd_utils": {
        "overview": (
            "This module offers a small interface for text-to-image generation with Stable Diffusion. It keeps "
            "credential retrieval, local pipeline setup, hosted inference, and image display behind one client so "
            "students can focus on prompt choices and outputs."
        ),
        "helps": (
            "retrieve a Hugging Face credential from the course token mechanism",
            "initialize either a local diffusion pipeline or hosted inference client",
            "generate multiple images from one prompt for comparison",
            "display and return outputs for notebook-based analysis",
        ),
        "uses": (
            "You are studying how prompt wording changes generated visual content.",
            "You need a consistent interface for local and hosted diffusion workflows.",
            "You understand that model download, inference time, and hosted usage may require external resources.",
        ),
        "workflow": (
            "Choose a model and local or hosted execution mode.",
            "Initialize ``SDClient`` with the required credential and cache settings.",
            "Generate a small set of images from a controlled prompt.",
            "Compare outputs and record prompt, model, and generation settings.",
        ),
        "starter_kits": (("Multimodal AI", "../../starter_kits/m3_mm.html"),),
    },
    "svi_utils": {
        "overview": (
            "This module supports urban computer-vision workflows that begin with Google Street View imagery. "
            "It can sample locations, check image availability, download views, and connect those images to "
            "segmentation and visualization steps."
        ),
        "helps": (
            "generate a coordinate grid for systematic street-view sampling",
            "check whether imagery is available before requesting a download",
            "download individual or batched street-view images with metadata",
            "run segmentation and visually compare source images with predicted classes",
        ),
        "uses": (
            "Your study links urban locations to visible streetscape characteristics.",
            "You need a repeatable sampling method rather than manually selecting screenshots.",
            "You have confirmed API terms, quota, credential, and privacy requirements before online collection.",
        ),
        "workflow": (
            "Define the study extent and generate candidate coordinates.",
            "Check availability and download a small validation sample.",
            "Run segmentation on the saved images.",
            "Inspect source/prediction pairs before summarizing class coverage.",
        ),
        "starter_kits": (("Computer Vision", "../../starter_kits/m4_cv.html"),),
    },
    "token_utils": {
        "overview": (
            "This module centralizes how course utilities obtain API credentials. It checks environment variables "
            "and the local token file so notebooks can request a named service token without embedding secrets in "
            "teaching material."
        ),
        "helps": (
            "look up a service credential through one consistent function",
            "keep notebook examples free of literal API keys",
            "optionally prompt for a missing token during an interactive session",
            "provide clearer errors when a required credential is unavailable",
        ),
        "uses": (
            "Another utility needs a DeepSeek, Hugging Face, or Google credential.",
            "A notebook should work across student machines without machine-specific paths.",
            "You need credential handling only; this module does not make the external API request itself.",
        ),
        "workflow": (
            "Store the credential in the supported local configuration or environment variable.",
            "Request it by service name with ``get_token``.",
            "Pass the returned value directly to the client that needs it.",
            "Never print, save, or commit the returned credential.",
        ),
        "starter_kits": (),
    },
    "viz_utils": {
        "overview": (
            "This module contains visualization helpers used across the course, from neural-network diagrams and "
            "training curves to review analysis and urban maps. The functions turn intermediate model or data "
            "results into figures that can be interpreted and communicated."
        ),
        "helps": (
            "draw simplified model structures and learning curves",
            "visualize distributions, bias comparisons, keywords, and co-occurrence patterns",
            "map sampled points, reviews, points of interest, and heat surfaces",
            "reduce repeated plotting setup while retaining interpretable inputs",
        ),
        "uses": (
            "A notebook has computed results but still needs an explanatory figure.",
            "You want consistent visual conventions across a starter kit or comparison.",
            "You will choose a plot based on the analytical question rather than merely on available columns.",
        ),
        "workflow": (
            "Identify the comparison, spatial pattern, or model behavior the figure should communicate.",
            "Prepare the function's expected DataFrame, array, model, or coordinates.",
            "Generate the plot and inspect labels, scale, and missing values.",
            "Add interpretation in the notebook instead of treating the visualization as the conclusion.",
        ),
        "starter_kits": (("Bias Detection and Interpretability", "../../starter_kits/m5_bias.html"),),
    },
    "yolo_utils": {
        "overview": (
            "This module supports an object-detection workflow based on YOLO predictions over images or video. "
            "It helps students move from frame-level detections to saved records and an annotated video that can "
            "be inspected or summarized."
        ),
        "helps": (
            "run object detection and tracking over a video source",
            "record detected classes, confidence values, and frame information",
            "apply lightweight smoothing to reduce unstable frame-to-frame results",
            "render an annotated output video for qualitative review",
        ),
        "uses": (
            "Your project asks what objects appear in a video and how detections change over time.",
            "You need both a tabular record for analysis and a visual output for validation.",
            "You will inspect false positives and missed detections before using counts as evidence.",
        ),
        "workflow": (
            "Select a YOLO model, video source, and classes relevant to the question.",
            "Run ``detect_and_track`` and save the frame-level results.",
            "Review confidence thresholds and temporal consistency.",
            "Create an annotated video with ``visualize_video`` and compare it with the table.",
        ),
        "starter_kits": (("Computer Vision", "../../starter_kits/m4_cv.html"),),
    },
}


@dataclass(frozen=True)
class ModuleAPI:
    """Public objects declared in one Python module."""

    functions: tuple[str, ...]
    classes: tuple[str, ...]
    methods: tuple[str, ...]
    constants: tuple[str, ...]


def inspect_module(path: Path) -> ModuleAPI:
    """Return public declarations from *path* without importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: list[str] = []
    classes: list[str] = []
    methods: list[str] = []
    constants: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    methods.append(f"{node.name}.{child.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                    constants.append(target.id)

    return ModuleAPI(tuple(functions), tuple(classes), tuple(methods), tuple(constants))


def _autosummary(title: str, objects: tuple[str, ...], *, module: str) -> str:
    if not objects:
        return ""
    lines = [title, "-" * len(title), "", ".. autosummary::", "   :toctree: ../generated", ""]
    lines.extend(f"   ~{module}.{name}" for name in objects)
    return "\n".join(lines) + "\n\n"


def render_module_page(module_name: str, api: ModuleAPI) -> str:
    """Render a module source page with categorized autosummary tables."""
    full_name = f"ccai9012.{module_name}"
    guide = MODULE_METADATA[module_name]
    lines = [
        module_name,
        "=" * len(module_name),
        "",
        guide["overview"],
        "",
        "What this module helps you do",
        "-----------------------------",
        "",
        *(f"* {item}" for item in guide["helps"]),
        "",
        "When to use it",
        "--------------",
        "",
        *(f"* {item}" for item in guide["uses"]),
        "",
        "Typical workflow",
        "----------------",
        "",
        *(f"#. {item}" for item in guide["workflow"]),
        "",
    ]
    starter_kits = guide["starter_kits"]
    if starter_kits:
        lines.extend(["Related Starter Kits", "--------------------", ""])
        lines.extend(f"* `{label} <{href}>`_" for label, href in starter_kits)
        lines.append("")

    page = "\n".join(lines) + "\n"
    page += _autosummary("Classes", api.classes, module=full_name)
    page += _autosummary("Functions", api.functions, module=full_name)

    if api.constants:
        page += "Constants\n---------\n\n"
        for constant in api.constants:
            page += f".. autodata:: {full_name}.{constant}\n\n"

    page += _autosummary("Methods", api.methods, module=full_name)
    return page.rstrip() + "\n"


def generate_module_sources() -> tuple[str, ...]:
    """Regenerate package/module reStructuredText sources."""
    module_dir = SOURCE_DIR / "ccai9012"
    generated_dir = SOURCE_DIR / "generated"
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    module_dir.mkdir(parents=True, exist_ok=True)
    modules = tuple(sorted(path.stem for path in PACKAGE_DIR.glob("*.py") if path.stem != "__init__"))

    index = [
        "Modules",
        "-------",
        "",
        "The package is organized by teaching task. Choose a module below, then open an",
        "individual object page for its complete signature, parameters, and return value.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    index.extend(f"   {name}" for name in modules)
    (module_dir / "index.rst").write_text("\n".join(index) + "\n", encoding="utf-8")

    for name in modules:
        api = inspect_module(PACKAGE_DIR / f"{name}.py")
        (module_dir / f"{name}.rst").write_text(render_module_page(name, api), encoding="utf-8")
    return modules


def build_api_docs() -> Path:
    """Run a clean Sphinx warning-as-error HTML build and return its output path."""
    generate_module_sources()
    sphinx_build = shutil.which("sphinx-build")
    if not sphinx_build:
        raise FileNotFoundError("sphinx-build is not available; install the documentation dependencies first")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    with tempfile.TemporaryDirectory(prefix="ccai9012-api-doctrees-") as doctree_dir:
        subprocess.run(
            [
                sphinx_build,
                "-W",
                "--keep-going",
                "-b",
                "html",
                "-d",
                doctree_dir,
                str(SOURCE_DIR),
                str(OUTPUT_DIR),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
        )
    # Sphinx/theme templates contain whitespace-only indentation lines. Normalize
    # generated text assets so committed GitHub Pages output passes diff checks.
    for suffix in ("*.html", "*.css", "*.js"):
        for path in OUTPUT_DIR.rglob(suffix):
            content = path.read_text(encoding="utf-8")
            normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
            if normalized != content:
                path.write_text(normalized, encoding="utf-8")
    return OUTPUT_DIR


def main() -> None:
    build_api_docs()


if __name__ == "__main__":
    main()

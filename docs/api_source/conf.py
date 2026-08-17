"""Sphinx configuration for the CCAI9012 API reference."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

project = "API Reference"
author = "CCAI9012"
copyright = "2026, CCAI9012"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_default_options = {"member-order": "bysource", "show-inheritance": True}
autodoc_member_order = "bysource"
add_module_names = False
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Importing ccai9012/__init__.py loads every course module. Mock optional runtime
# dependencies so documentation can build without downloading models or tokens.
autodoc_mock_imports = [
    "branca",
    "cv2",
    "diffusers",
    "folium",
    "geopandas",
    "huggingface_hub",
    "IPython",
    "langchain",
    "langchain_community",
    "langchain_deepseek",
    "matplotlib",
    "numpy",
    "osmnx",
    "pandas",
    "PIL",
    "plotly",
    "pygraphviz",
    "pyproj",
    "qwen_vl_utils",
    "requests",
    "scipy",
    "seaborn",
    "shapely",
    "sklearn",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
    "wordcloud",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"
root_doc = "index"
language = "en"

html_theme = "pydata_sphinx_theme"
html_title = "CCAI9012 API Reference"
html_static_path = ["_static"]
html_css_files = ["api-theme.css"]
html_theme_options = {
    "show_toc_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "show_nav_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav.html", "course-home-link.html"],
    "navbar_persistent": [],
    "navbar_end": ["search-field.html", "theme-switcher.html"],
}
html_sidebars = {"**": ["api-sidebar.html"]}
html_context = {
    "github_user": "ccai9012",
    "github_repo": "ccai9012",
    "github_version": "main",
    "doc_path": "docs/api_source",
}
html_show_sourcelink = True
html_copy_source = False

# Linkcheck focuses on repository-owned navigation. External services are not
# required for an offline documentation build.
linkcheck_ignore = [r"https://.*"]


def _public_api_members(module_name: str) -> dict[str, list[dict[str, str]]]:
    """Collect the documented public API without importing course modules."""
    module_path = REPO_ROOT / "ccai9012" / f"{module_name}.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    members = {"Classes": [], "Functions": [], "Methods": []}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            members["Classes"].append({"label": node.name, "object": node.name})
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    qualified_name = f"{node.name}.{child.name}"
                    members["Methods"].append({"label": qualified_name, "object": qualified_name})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            members["Functions"].append({"label": node.name, "object": node.name})

    return members


def _set_course_home_href(app, pagename, templatename, context, doctree) -> None:
    """Expose depth-correct course and API navigation links."""
    context["course_home_href"] = "../" * (pagename.count("/") + 1) + "index.html"
    context["api_modules"] = [
        "gan_utils",
        "llm_utils",
        "multi_modal_utils",
        "nn_utils",
        "sd_utils",
        "svi_utils",
        "token_utils",
        "viz_utils",
        "yolo_utils",
    ]
    context["api_members"] = {
        module_name: _public_api_members(module_name)
        for module_name in context["api_modules"]
    }
    context["active_api_object"] = ""
    if pagename == "index":
        context["api_home_href"] = "index.html"
        context["api_module_prefix"] = "ccai9012/"
        context["api_object_prefix"] = "generated/ccai9012."
        context["active_api_module"] = ""
    elif pagename.startswith("ccai9012/"):
        context["api_home_href"] = "../index.html"
        context["api_module_prefix"] = ""
        context["api_object_prefix"] = "../generated/ccai9012."
        context["active_api_module"] = pagename.split("/", 1)[1]
    else:
        context["api_home_href"] = "../index.html"
        context["api_module_prefix"] = "../ccai9012/"
        context["api_object_prefix"] = "ccai9012."
        filename = pagename.rsplit("/", 1)[-1]
        parts = filename.split(".")
        context["active_api_module"] = parts[1] if len(parts) > 2 and parts[0] == "ccai9012" else ""
        if context["active_api_module"]:
            context["active_api_object"] = ".".join(parts[2:])


def setup(app):
    """Register API-theme page context values."""
    app.connect("html-page-context", _set_course_home_href)

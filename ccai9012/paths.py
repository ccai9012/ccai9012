"""Stable paths for resources shipped with the CCAI9012 repository.

The constants are derived from this module's location, so notebooks can be
run from the repository root, their own directory, or another current working
directory without changing resource resolution.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent

STARTER_KITS_DIR = REPOSITORY_ROOT / "starter_kits"
WEEKLY_SCRIPTS_DIR = REPOSITORY_ROOT / "weekly_scripts"
DATA_DIR = REPOSITORY_ROOT / "data"
MODELS_DIR = REPOSITORY_ROOT / "models"
CACHE_DIR = REPOSITORY_ROOT / "cache"
OUTPUT_DIR = REPOSITORY_ROOT / "output"


def ensure_output_dir(example_dir: str | Path, name: str = "output") -> Path:
    """Create and return an example-local output directory.

    Parameters
    ----------
    example_dir:
        Directory belonging to the notebook or example.
    name:
        Output directory name relative to ``example_dir``.
    """

    output_path = Path(example_dir).expanduser().resolve() / name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


__all__ = [
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

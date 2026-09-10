"""Focused tests for repository path resolution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ccai9012


def test_public_paths_are_absolute_paths() -> None:
    for name in (
        "PACKAGE_ROOT",
        "REPOSITORY_ROOT",
        "STARTER_KITS_DIR",
        "WEEKLY_SCRIPTS_DIR",
        "DATA_DIR",
        "MODELS_DIR",
        "CACHE_DIR",
        "OUTPUT_DIR",
    ):
        value = getattr(ccai9012, name)
        assert isinstance(value, Path)
        assert value.is_absolute()


def test_paths_resolve_from_module_location() -> None:
    assert ccai9012.PACKAGE_ROOT == Path(ccai9012.__file__).resolve().parent
    assert ccai9012.REPOSITORY_ROOT == ccai9012.PACKAGE_ROOT.parent
    assert ccai9012.STARTER_KITS_DIR == ccai9012.REPOSITORY_ROOT / "starter_kits"


def test_import_does_not_create_directories(tmp_path: Path) -> None:
    script = "import ccai9012; print(ccai9012.REPOSITORY_ROOT)"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ccai9012.REPOSITORY_ROOT)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert list(tmp_path.iterdir()) == []


def test_ensure_output_dir_is_explicit(tmp_path: Path) -> None:
    output_path = ccai9012.ensure_output_dir(tmp_path, "teaching_output")
    assert output_path == (tmp_path / "teaching_output").resolve()
    assert output_path.is_dir()

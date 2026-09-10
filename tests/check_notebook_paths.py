"""Audit scoped notebook code cells for fragile resource paths.

Markdown links may remain relative because they are resolved by the notebook
or generated site. Code-cell data/model/cache paths must use the package path
layer when they cross out of the notebook's own example directory.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCOPED_NOTEBOOKS = (
    "starter_kits/1_traditional_generative_ml/GANmapper/biulding_profile_gen.ipynb",
    "weekly_scripts/wip/week6/week6_t_gan.ipynb",
    "weekly_scripts/wip/week7/week7_t_doc_parsing.ipynb",
    "starter_kits/3_multimodal_reasoning/gen_images_eval/gen_image_eval.ipynb",
    "starter_kits/4_cv_models/svi_housing_price/prediction_svi.ipynb",
    "starter_kits/4_cv_models/webcam_yolo/pedestrain_yolo.ipynb",
    "weekly_scripts/wip/week8/svi_segmentation/week8_t_svi.ipynb",
    "weekly_scripts/wip/week8/yolo/week8_t_yolo.ipynb",
)

FRAGILE_RESOURCE_PATH = re.compile(
    r"['\"](?P<path>(?:\.\./)+(?:data|models?|cache)(?:/|['\"]))"
)


def audit_notebook_paths(paths: tuple[str, ...] = SCOPED_NOTEBOOKS) -> list[str]:
    """Return one diagnostic for each fragile code-cell resource path."""

    diagnostics: list[str] = []
    for relative_path in paths:
        notebook_path = REPOSITORY_ROOT / relative_path
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell_number, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as exc:
                diagnostics.append(f"{relative_path}:cell {cell_number}: AST error: {exc}")
                continue
            for match in FRAGILE_RESOURCE_PATH.finditer(source):
                diagnostics.append(
                    f"{relative_path}:cell {cell_number}: fragile resource path {match.group('path')}"
                )
    return diagnostics


if __name__ == "__main__":
    problems = audit_notebook_paths()
    print(f"notebooks_checked={len(SCOPED_NOTEBOOKS)}")
    print(f"path_audit_errors={len(problems)}")
    for problem in problems:
        print(problem)
    raise SystemExit(1 if problems else 0)

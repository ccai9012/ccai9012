"""Offline contract checks for Module 2 sample data and cached schemas."""

import json
from pathlib import Path

import pandas as pd

from ccai9012 import llm_utils


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "starter_kits/2_llm_structure_output"


def test_module2_manifest_and_sample_files() -> None:
    manifest = json.loads((MODULE_ROOT / "sample_manifest.json").read_text())
    for config in manifest["airbnb"]["configs"]:
        reviews = pd.read_csv(ROOT / config["reviews"])
        listings = pd.read_csv(ROOT / config["listings"])
        assert len(reviews) == config["review_rows"]
        assert len(listings) == config["listing_rows"]
        assert set(reviews.columns) == set(manifest["airbnb"]["schema"]["reviews"])
    urban = pd.read_csv(ROOT / manifest["urban_sentiment"]["path"])
    assert len(urban) == manifest["urban_sentiment"]["rows"]
    assert set(urban.columns) == set(manifest["urban_sentiment"]["schema"])


def test_cached_literature_table_is_parseable() -> None:
    cached = pd.read_csv(MODULE_ROOT / "lit_review/output/multiple_comparison.csv")
    parsed = llm_utils.parse_markdown_table(str(cached.iloc[0]["extracted_text"]))
    assert {"Problem", "Research Gap", "Methodology", "Key Results"} <= set(parsed.columns)
    assert not parsed.empty

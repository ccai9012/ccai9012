"""Data preparation for the Airbnb review lesson."""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd

from .paths import REPOSITORY_ROOT, STARTER_KITS_DIR


def load_airbnb_data(
    data_source: str = "sample", config_name: str = "central_western"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Load one Hong Kong review configuration and its map boundaries.

    The sample mode uses small files tracked in the repository. Full mode reads
    the original compressed Inside Airbnb files and limits reviews to the
    selected neighbourhood. Neither mode calls an external service.

    Args:
        data_source: ``"sample"`` or ``"full"``.
        config_name: ``"central_western"`` or ``"yau_tsim_mong"``.

    Returns:
        Reviews, listings, their joined rows, and neighbourhood polygons.

    Raises:
        ValueError: If the source or configuration is unknown, required
            columns are missing, or a review cannot be joined to a listing.
    """
    module_root = STARTER_KITS_DIR / "2_llm_structure_output"
    example_root = module_root / "airbnb_reviews"
    manifest = json.loads(
        (module_root / "sample_manifest.json").read_text(encoding="utf-8")
    )
    configs = {item["name"]: item for item in manifest["airbnb"]["configs"]}
    if config_name not in configs:
        raise ValueError(f"Unknown configuration {config_name!r}; choose from {sorted(configs)}")
    config = configs[config_name]

    if data_source == "sample":
        reviews = pd.read_csv(REPOSITORY_ROOT / config["reviews"])
        listings = pd.read_csv(REPOSITORY_ROOT / config["listings"])
    elif data_source == "full":
        reviews = pd.read_csv(example_root / "data/reviews.csv.gz", compression="gzip")
        listings = pd.read_csv(example_root / "data/listings.csv.gz", compression="gzip")
        listing_ids = listings.loc[
            listings["neighbourhood_cleansed"].eq(config["neighbourhood"]), "id"
        ]
        reviews = reviews[reviews["listing_id"].isin(listing_ids)].head(500)
    else:
        raise ValueError("data_source must be 'sample' or 'full'")

    required_review = {"listing_id", "id", "date", "comments"}
    required_listing = {
        "id", "name", "latitude", "longitude", "neighbourhood_cleansed",
        "review_scores_rating",
    }
    if required_review - set(reviews.columns) or required_listing - set(listings.columns):
        raise ValueError("Airbnb input files are missing required review or listing columns")
    joined = reviews.merge(
        listings, left_on="listing_id", right_on="id", how="left",
        suffixes=("_review", "_listing"), validate="many_to_one",
    )
    if joined[["latitude", "longitude"]].isna().any().any():
        raise ValueError("At least one review lacks a listing location")
    neighbourhoods = gpd.read_file(example_root / "data/neighbourhoods.geojson")
    return reviews, listings, joined, neighbourhoods

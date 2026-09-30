"""Repository-local data adapter for the German credit fairness lesson."""

from pathlib import Path

import pandas as pd


GERMAN_COLUMNS = [
    "status", "month", "credit_history", "purpose", "credit_amount", "savings",
    "employment", "investment_as_income_percentage", "personal_status",
    "other_debtors", "residence_since", "property", "age", "installment_plans",
    "housing", "number_of_credits", "skill_level", "people_liable_for",
    "telephone", "foreign_worker", "credit",
]


def load_german_credit_dataset(data_path):
    """Load tracked German credit data into an AIF360 dataset without copying files."""
    from aif360.datasets import StandardDataset
    from aif360.datasets.german_dataset import default_preprocessing

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    if len(frame) != 1000:
        raise ValueError(f"Expected 1000 German credit records, found {len(frame)}")
    return StandardDataset(
        df=frame, label_name="credit", favorable_classes=lambda label: label == 1,
        protected_attribute_names=["age"], privileged_classes=[lambda age: age >= 25],
        categorical_features=[
            "status", "credit_history", "purpose", "savings", "employment",
            "other_debtors", "property", "installment_plans", "housing",
            "skill_level", "telephone", "foreign_worker",
        ],
        features_to_drop=["personal_status", "sex"],
        custom_preprocessing=default_preprocessing,
        metadata={
            "label_maps": [{1.0: "Good Credit", 0.0: "Bad Credit"}],
            "protected_attribute_maps": [{1.0: "Age 25+", 0.0: "Under 25"}],
        },
    )

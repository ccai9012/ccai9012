"""Focused checks for the manifest-driven GANmapper data path."""

from pathlib import Path

from ccai9012 import gan_utils


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GANMAPPER_DATA = REPOSITORY_ROOT / "starter_kits/1_traditional_generative_ml/GANmapper/data/Exp4"


def test_pair_manifest_is_deterministic_and_aligned() -> None:
    first = gan_utils.build_pair_manifest(GANMAPPER_DATA, max_pairs=12, random_seed=9012)
    second = gan_utils.build_pair_manifest(GANMAPPER_DATA, max_pairs=12, random_seed=9012)

    assert first == second
    assert len(first) == 12
    for pair in first:
        source = GANMAPPER_DATA / pair["source"]
        target = GANMAPPER_DATA / pair["target"]
        assert source.is_file()
        assert target.is_file()
        assert pair["relative_path"] in pair["source"]
        assert pair["relative_path"] in pair["target"]


def test_pair_split_does_not_change_global_random_state() -> None:
    pairs = gan_utils.build_pair_manifest(GANMAPPER_DATA, max_pairs=4, random_seed=9012)
    import random

    random.seed(123)
    expected = random.random()
    random.seed(123)
    gan_utils.split_pairs(pairs, random_seed=7)
    assert random.random() == expected

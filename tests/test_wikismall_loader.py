from pathlib import Path

import pytest

from data.wikismall_loader import WikiSmallLoader, load_raw_wikismall_split
from prompts import simplify_prompt


def test_raw_wikismall_split_counts_match_checked_in_files() -> None:
    assert len(load_raw_wikismall_split("train")) == 88837
    assert len(load_raw_wikismall_split("valid")) == 205
    assert len(load_raw_wikismall_split("test")) == 100


def test_wikismall_loader_uses_complex_source_and_simple_target() -> None:
    loader = WikiSmallLoader(max_train_samples=1, max_eval_samples=1, seed=42)

    train_pairs, valid_pairs, test_pairs = loader.load_pairs()

    assert len(train_pairs) == 1
    assert len(valid_pairs) == 1
    assert len(test_pairs) == 1
    assert valid_pairs[0] == (
        simplify_prompt(
            "The program was transmitted by Onda Cero Radio until July 2007 , "
            "when the last program was broadcasted ."
        ),
        "Onda Cero .",
    )
    assert test_pairs[0] == (
        simplify_prompt(
            "Genetic engineering has expanded the genes available to breeders to utilize "
            "in creating desired germlines for new crops ."
        ),
        "New plants were created with genetic engineering .",
    )


def test_wikismall_split_rejects_mismatched_line_counts(tmp_path: Path) -> None:
    (tmp_path / "PWKP_108016.tag.80.aner.ori.train.src").write_text(
        "complex one\ncomplex two\n",
        encoding="utf-8",
    )
    (tmp_path / "PWKP_108016.tag.80.aner.ori.train.dst").write_text(
        "simple one\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Mismatched WikiSmall/train line counts"):
        load_raw_wikismall_split("train", tmp_path)

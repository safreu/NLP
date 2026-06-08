import pytest

from metrics.metric_sari import compute_sari, format_references


def test_format_references_wraps_single_strings() -> None:
    assert format_references(["a reference"]) == [["a reference"]]


def test_format_references_keeps_lists_of_references() -> None:
    references = format_references([["ref one", "ref two"]])

    assert references == [["ref one", "ref two"]]


def test_format_references_coerces_items_to_strings() -> None:
    assert format_references([[1, 2]]) == [["1", "2"]]


def test_compute_sari_returns_score_in_expected_range() -> None:
    # The SARI metric script is downloaded from the Hugging Face Hub on first
    # use, so skip this smoke test when it cannot be loaded.
    try:
        score = compute_sari(
            ["The physician administered medication to the patient."],
            ["The doctor gave medicine to the patient."],
            [
                [
                    "The doctor gave medicine to the patient.",
                    "The doctor gave the patient medicine.",
                ]
            ],
        )
    except Exception as error: 
        pytest.skip(f"SARI metric unavailable: {error}")

    assert 0.0 <= score <= 100.0

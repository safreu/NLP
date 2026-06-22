from evaluation.number_preservation import (
    compare_numbers,
    detect_columns,
    extract_numbers,
    normalize_number,
)


def test_normalize_number_handles_spacing_commas_decimals_and_percent() -> None:
    assert normalize_number("1, 250") == "1250"
    assert normalize_number("42.0") == "42"
    assert normalize_number("12 . 50 %") == "12.5%"


def test_extract_numbers_preserves_signed_and_percent_values() -> None:
    assert extract_numbers("Revenue fell -3.0% from 1,200 to 950.") == [
        "-3%",
        "1200",
        "950",
    ]


def test_compare_numbers_counts_duplicate_losses() -> None:
    comparison = compare_numbers(
        ["2024", "2024", "12%"],
        ["2024"],
        "The output only kept 2024.",
    )

    assert comparison["preserved_numbers"] == 1
    assert comparison["lost_numbers"] == 2
    assert comparison["lost_number_values"] == ["2024", "12%"]
    assert comparison["has_number_loss"] is True


def test_detect_columns_accepts_project_prediction_aliases() -> None:
    assert detect_columns(["source_sentence", "predicted_sentence", "reference"]) == (
        "source_sentence",
        "predicted_sentence",
    )

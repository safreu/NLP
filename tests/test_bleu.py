import pytest

from metrics.Bleu import compute_bleuscore


def test_compute_bleuscore_for_perfect_match() -> None:
    score = compute_bleuscore(
        ["the cat sat on the mat"],
        ["the cat sat on the mat"],
    )

    assert score == 1.0


def test_compute_bleuscore_for_no_overlap_is_low() -> None:
    score = compute_bleuscore(
        ["totally different words here"],
        ["the cat sat on the mat"],
    )

    assert 0.0 <= score < 0.1


def test_compute_bleuscore_returns_value_in_unit_range() -> None:
    score = compute_bleuscore(
        ["the doctor gave the patient medicine"],
        ["the doctor gave medicine to the patient"],
    )

    assert 0.0 <= score <= 1.0


def test_compute_bleuscore_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        compute_bleuscore(["one candidate"], [])

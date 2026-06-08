import pytest

from metrics.rouge import compute_rougescore


def test_compute_rougescore_for_perfect_match() -> None:
    score = compute_rougescore(
        ["the cat sat on the mat"],
        ["the cat sat on the mat"],
    )

    assert score == pytest.approx(1.0)


def test_compute_rougescore_for_partial_overlap() -> None:
    score = compute_rougescore(
        ["the cat sat"],
        ["the cat sat on the mat"],
    )

    assert score == pytest.approx(2 / 3)


def test_compute_rougescore_averages_over_pairs() -> None:
    score = compute_rougescore(
        ["the cat sat on the mat", "the cat sat"],
        ["the cat sat on the mat", "the cat sat on the mat"],
    )

    assert score == pytest.approx((1.0 + 2 / 3) / 2)


def test_compute_rougescore_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        compute_rougescore(["one prediction"], [])

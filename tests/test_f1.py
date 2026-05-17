import pytest

from metrics.f1 import compute_f1, compute_token_f1_score, tokenize


def test_tokenize_lowercases_and_removes_punctuation() -> None:
    assert tokenize("The Doctor gave medicine!") == ["the", "doctor", "gave", "medicine"]


def test_compute_token_f1_score_for_perfect_match() -> None:
    precision, recall, f1 = compute_token_f1_score(
        "The doctor gave medicine.",
        "The doctor gave medicine.",
    )

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_compute_token_f1_score_for_partial_match() -> None:
    precision, recall, f1 = compute_token_f1_score(
        "the doctor gave medicine",
        "the doctor gave the patient medicine",
    )

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(4 / 6)
    assert f1 == pytest.approx(0.8)


def test_compute_token_f1_score_counts_repeated_words_correctly() -> None:
    precision, recall, f1 = compute_token_f1_score(
        "very very simple",
        "very simple",
    )

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(0.8)


def test_compute_f1() -> None:
    predictions = [
        "The doctor gave medicine.",
        "The cat sat on the mat.",
    ]
    references = [
        "The doctor gave medicine.",
        "A cat is sitting on the mat.",
    ]

    result = compute_f1(predictions, references)

    assert len(result["precision"]) == 2
    assert len(result["recall"]) == 2
    assert len(result["f1"]) == 2
    assert result["f1_mean"] == pytest.approx(sum(result["f1"]) / 2)


def test_compute_f1_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        compute_f1(["One prediction."], [])

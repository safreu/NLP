import pytest

from metrics.flesch_kincaid import (
    compute_flesch_kincaid,
    compute_flesch_kincaid_score,
    count_sentences,
    count_syllables_in_word,
    count_words,
)


def test_count_sentences() -> None:
    assert count_sentences("This is one sentence.") == 1
    assert count_sentences("This is one. This is two!") == 2
    assert count_sentences("This has no punctuation") == 1


def test_count_words() -> None:
    assert count_words("The doctor gave medicine to the patient.") == 7


def test_count_syllables_in_word() -> None:
    assert count_syllables_in_word("cat") == 1
    assert count_syllables_in_word("medicine") == 3
    assert count_syllables_in_word("make") == 1


def test_compute_flesch_kincaid_score() -> None:
    score = compute_flesch_kincaid_score("The cat sat on the mat.")

    assert score == pytest.approx(-1.45)


def test_compute_flesch_kincaid() -> None:
    predictions = [
        "The cat sat on the mat.",
        "The doctor gave medicine to the patient.",
    ]
    references = [
        "A cat is sitting on a mat.",
        "The physician administered medication to the patient.",
    ]

    result = compute_flesch_kincaid(predictions, references)

    assert len(result["scores"]) == 2
    assert result["mean"] == pytest.approx(sum(result["scores"]) / 2)


def test_compute_flesch_kincaid_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        compute_flesch_kincaid(["One prediction."], [])

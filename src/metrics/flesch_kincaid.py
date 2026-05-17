import re
from collections.abc import Sequence
from typing import TypedDict


class FleschKincaidResult(TypedDict):
    scores: list[float]
    mean: float


def count_sentences(text: str) -> int:
    # Count common sentence-ending punctuation marks.
    sentence_endings = re.findall(r"[.!?]+", text)

    # If there is text but no punctuation, treat it as one sentence.
    if not sentence_endings and count_words(text) > 0:
        return 1

    return len(sentence_endings)


def count_words(text: str) -> int:
    # Match simple English words and contractions such as "don't".
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return len(words)


def count_syllables_in_word(word: str) -> int:
    # This is an estimate: English syllable counting has many exceptions.
    cleaned_word = re.sub(r"[^a-z]", "", word.lower())

    if not cleaned_word:
        return 0

    vowel_groups = re.findall(r"[aeiouy]+", cleaned_word)
    syllable_count = len(vowel_groups)

    # A silent final "e" usually does not add a syllable, as in "make".
    if cleaned_word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # Every real word should count as at least one syllable.
    return max(1, syllable_count)


def count_syllables(text: str) -> int:
    # Reuse the same word matching rule so word and syllable counts agree.
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return sum(count_syllables_in_word(word) for word in words)


def compute_flesch_kincaid_score(text: str) -> float:
    # FKGL = 0.39 * words per sentence + 11.8 * syllables per word - 15.59
    # Lower scores usually mean the text is easier to read.
    sentence_count = count_sentences(text)
    word_count = count_words(text)
    syllable_count = count_syllables(text)

    if sentence_count == 0 or word_count == 0:
        return 0.0

    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count

    return (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59


def compute_flesch_kincaid(
    predictions: Sequence[str],
    references: Sequence[str],
) -> FleschKincaidResult:
    # The reference list is accepted so this matches other metric functions.
    # FKGL only measures readability, so it calculates scores from predictions.
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions length ({len(predictions)}) and references length "
            f"({len(references)}) do not match."
        )

    scores = [compute_flesch_kincaid_score(prediction) for prediction in predictions]

    if not scores:
        return {"scores": [], "mean": 0.0}

    return {"scores": scores, "mean": sum(scores) / len(scores)}


if __name__ == "__main__":
    predictions = [
        "The doctor gave medicine to the patient.",
        "The physician administered medication to the patient.",
    ]

    references = [
        "The doctor gave the patient medicine.",
        "The doctor gave medicine to the patient.",
    ]

    result = compute_flesch_kincaid(predictions, references)

    for prediction, score in zip(predictions, result["scores"], strict=True):
        print(f"Prediction: {prediction}")
        print(f"Flesch-Kincaid Grade Level: {score:.2f}")

    print(f"Mean Flesch-Kincaid Grade Level: {result['mean']:.2f}")

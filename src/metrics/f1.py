import re
from collections import Counter
from collections.abc import Sequence
from typing import TypedDict


class F1Result(TypedDict):
    precision: list[float]
    recall: list[float]
    f1: list[float]
    precision_mean: float
    recall_mean: float
    f1_mean: float


def tokenize(text: str) -> list[str]:
    # Convert text to lowercase so "Doctor" and "doctor" count as the same word.
    lowercase_text = text.lower()

    # Extract word tokens and ignore punctuation.
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", lowercase_text)


def compute_token_f1_score(prediction: str, reference: str) -> tuple[float, float, float]:
    # Token-level F1 compares the words in the prediction with the words in the reference.
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)

    if not prediction_tokens and not reference_tokens:
        return 1.0, 1.0, 1.0

    if not prediction_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0

    # Counter keeps track of how many times each token appears.
    prediction_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)

    # The & operator keeps only the overlapping tokens with the smaller count.
    overlap_counts = prediction_counts & reference_counts
    overlap = sum(overlap_counts.values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_f1(predictions: Sequence[str], references: Sequence[str]) -> F1Result:
    # Each prediction should have one matching reference sentence.
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions length ({len(predictions)}) and references length "
            f"({len(references)}) do not match."
        )

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []

    for prediction, reference in zip(predictions, references, strict=True):
        precision, recall, f1 = compute_token_f1_score(prediction, reference)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    if not f1_scores:
        return {
            "precision": [],
            "recall": [],
            "f1": [],
            "precision_mean": 0.0,
            "recall_mean": 0.0,
            "f1_mean": 0.0,
        }

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "f1": f1_scores,
        "precision_mean": sum(precision_scores) / len(precision_scores),
        "recall_mean": sum(recall_scores) / len(recall_scores),
        "f1_mean": sum(f1_scores) / len(f1_scores),
    }


if __name__ == "__main__":
    predictions = [
        "The doctor gave medicine to the patient.",
        "The cat sat on the mat.",
    ]

    references = [
        "The doctor gave the patient medicine.",
        "A cat is sitting on the mat.",
    ]

    result = compute_f1(predictions, references)

    for prediction, reference, f1_score in zip(
        predictions,
        references,
        result["f1"],
        strict=True,
    ):
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        print(f"Token-level F1: {f1_score:.2f}")

    print(f"Mean token-level F1: {result['f1_mean']:.2f}")

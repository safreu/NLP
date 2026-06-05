from collections.abc import Sequence
from typing import Any

from evaluate import load

_sari_metric: Any | None = None


def get_sari_metric() -> Any:
    global _sari_metric

    if _sari_metric is None:
        _sari_metric = load("sari")

    return _sari_metric


def format_references(references: Sequence[str | Sequence[str]]) -> list[list[str]]:
    formatted_references: list[list[str]] = []

    for reference in references:
        if isinstance(reference, str):
            formatted_references.append([reference])
        else:
            formatted_references.append([str(item) for item in reference])

    return formatted_references


def compute_sari(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str | Sequence[str]],
) -> float:
    result = get_sari_metric().compute(
        sources=list(sources),
        predictions=list(predictions),
        references=format_references(references),
    )

    return float(result["sari"])


def main() -> None:
    # Original complex sentence
    sources = ["The physician administered medication to the patient."]

    # System-generated simplification
    predictions = ["The doctor gave medicine to the patient."]

    # Human reference simplifications
    # Important: references must be a list of lists.
    references = [
        [
            "The doctor gave medicine to the patient.",
            "The doctor gave the patient medicine.",
        ]
    ]

    result = get_sari_metric().compute(
        sources=sources,
        predictions=predictions,
        references=references,
    )

    print(result)
    print(f"SARI score: {result['sari']:.2f}")


if __name__ == "__main__":
    main()

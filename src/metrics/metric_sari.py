from collections.abc import Sequence

from evaluate import load

sari = load("sari")


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
    result = sari.compute(
        sources=list(sources),
        predictions=list(predictions),
        references=format_references(references),
    )

    return float(result["sari"])


def main() -> None:
    # Load SARI metric
    sari = load("sari")

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

    result = sari.compute(sources=sources, predictions=predictions, references=references)

    print(result)
    print(f"SARI score: {result['sari']:.2f}")


if __name__ == "__main__":
    main()

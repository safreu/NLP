from evaluate import load


def main() -> None:
    # Load SARI metric
    sari = load("sari")

    # Original complex sentence
    sources = [
        "The physician administered medication to the patient."
    ]

    # System-generated simplification
    predictions = [
        "The doctor gave medicine to the patient."
    ]

    # Human reference simplifications
    # Important: references must be a list of lists.
    references = [
        [
            "The doctor gave medicine to the patient.",
            "The doctor gave the patient medicine."
        ]
    ]

    result = sari.compute(
        sources=sources,
        predictions=predictions,
        references=references
    )

    print(result)
    print(f"SARI score: {result['sari']:.2f}")


if __name__ == "__main__":
    main()
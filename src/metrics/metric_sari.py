from evaluate import load

sari = load("sari")

def compute_sari(sources: list[str], predictions: list[str], references: list[str]) -> float:
    f_references = [
        [reference] for reference in references
    ]
    
    result = sari.compute(
        sources=sources,
        predictions=predictions,
        references=f_references
    )
    
    return float(result["sari"])

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
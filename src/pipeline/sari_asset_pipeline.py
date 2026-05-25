from __future__ import annotations

from collections.abc import Sequence

from datasets import load_dataset
from evaluate import load
from transformers import pipeline

DATASET_NAME = "facebook/asset"
DATASET_CONFIG = "simplification"
SPLIT = "validation"
ORIGINAL_COLUMN = "original"
SIMPLIFICATIONS_COLUMN = "simplifications"

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_EXAMPLES = 5
MAX_NEW_TOKENS = 80
PROMPT_PREFIX = "Rewrite this sentence in simpler English. Return only the simplified sentence: "


def load_asset_examples(max_examples: int = MAX_EXAMPLES) -> tuple[list[str], list[list[str]]]:
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=f"{SPLIT}[:{max_examples}]",
    )

    sources: list[str] = []
    references: list[list[str]] = []

    for row in dataset:
        sources.append(str(row[ORIGINAL_COLUMN]))
        references.append([str(reference) for reference in row[SIMPLIFICATIONS_COLUMN]])

    return sources, references


def simplify_sentences(sources: Sequence[str], model_name: str = MODEL_NAME) -> list[str]:
    simplifier = pipeline(
        "text-generation",
        model=model_name,
    )

    predictions: list[str] = []

    for source in sources:
        result = simplifier(
            [{"role": "user", "content": f"{PROMPT_PREFIX}{source}"}],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
        )
        predictions.append(extract_generated_text(result))

    return predictions


def extract_generated_text(result: object) -> str:
    if not isinstance(result, list) or not result:
        raise RuntimeError(f"Unexpected pipeline result: {result!r}")

    first_result = result[0]
    if not isinstance(first_result, dict):
        raise RuntimeError(f"Unexpected pipeline result item: {first_result!r}")

    generated_text = first_result.get("generated_text")
    if not isinstance(generated_text, str):
        raise RuntimeError(f"Missing generated text in pipeline result: {first_result!r}")

    return generated_text.strip()


def compute_sari(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    sari = load("sari")
    result = sari.compute(
        sources=list(sources),
        predictions=list(predictions),
        references=[list(reference_set) for reference_set in references],
    )

    score = result["sari"]
    if not isinstance(score, int | float):
        raise RuntimeError(f"Unexpected SARI score: {score!r}")

    return float(score)


def print_examples(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> None:
    for index, (source, prediction, reference_set) in enumerate(
        zip(sources, predictions, references, strict=True),
        start=1,
    ):
        print(f"\nExample {index}")
        print(f"Source: {source}")
        print(f"Prediction: {prediction}")
        print("References:")
        for reference_index, reference in enumerate(reference_set[:3], start=1):
            print(f"  {reference_index}. {reference}")


def main() -> None:
    sources, references = load_asset_examples()
    predictions = simplify_sentences(sources)
    sari_score = compute_sari(sources, predictions, references)

    print_examples(sources, predictions, references)
    print(f"\nSARI on {len(sources)} ASSET validation examples: {sari_score:.2f}")


if __name__ == "__main__":
    main()

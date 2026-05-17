from typing import Any

from datasets import load_dataset


DATASET_NAME = "facebook/asset"
DATASET_CONFIG = "simplification"
ORIGINAL_COLUMN = "original"
SIMPLIFICATIONS_COLUMN = "simplifications"


def word_count(sentence: str) -> int:
    return len(sentence.split())


def asset_rows(split: Any) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []

    for data in split:
        original = str(data[ORIGINAL_COLUMN])
        simplifications = [str(sentence) for sentence in data[SIMPLIFICATIONS_COLUMN]]
        rows.append((original, simplifications))

    return rows


def average_lengths(split: Any) -> tuple[float, float]:
    original_sum = 0
    simplification_sum = 0
    simplification_count = 0

    rows = asset_rows(split)
    for original, simplifications in rows:
        original_sum += word_count(original)

        for simplification in simplifications:
            simplification_sum += word_count(simplification)
            simplification_count += 1

    original_average = original_sum / len(rows) if rows else 0.0
    simplification_average = (
        simplification_sum / simplification_count if simplification_count else 0.0
    )

    return original_average, simplification_average


def print_split_example(split_name: str, split: Any) -> None:
    if len(split) == 0:
        print(f"   {split_name} sample: no entries")
        return

    sample = split[min(1, len(split) - 1)]

    print(f"   {split_name} sample: ")
    print(f"    Original: {sample[ORIGINAL_COLUMN]}")
    for index, simplification in enumerate(sample[SIMPLIFICATIONS_COLUMN][:3], start=1):
        print(f"    Simplification {index}: {simplification}")
    print()


def main() -> None:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    print("Dataset structure: ")
    print(dataset)
    print()

    print("Features: ")
    for split_name in dataset:
        print(f"    {split_name}: {dataset[split_name].features}")
    print()

    print("Split sizes: ")
    for split_name in dataset:
        print(f"    {split_name}: {len(dataset[split_name])} entries")
    print()

    print("Average sentence length: ")
    for split_name in dataset:
        original_average, simplification_average = average_lengths(dataset[split_name])
        print(f"    {split_name} original: {original_average:.2f} words")
        print(f"    {split_name} simplifications: {simplification_average:.2f} words")
    print()

    print("Examples: ")
    for split_name in dataset:
        print_split_example(split_name, dataset[split_name])


if __name__ == "__main__":
    main()

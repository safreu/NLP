from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from datasets import load_dataset

from metrics.metric_sari import compute_sari
from storage.json_store import write_json


class AssetSariEvaluator:
    def __init__(
        self,
        split: str = "validation",
        max_examples: int = 0,
    ):
        self.split = split
        self.max_examples = max_examples

    def _load_asset(self) -> tuple[list[str], list[list[str]]]:
        split_name = self.split if self.max_examples <= 0 else f"{self.split}[:{self.max_examples}]"
        dataset = load_dataset("facebook/asset", "simplification", split=split_name)

        sources = []
        references = []

        for row in dataset:
            sources.append(str(row["original"]))
            references.append([str(ref) for ref in row["simplifications"]])

        return sources, references

    def run(
        self,
        predict_fn: Callable[[list[str]], list[str]],
        output_dir: Path,
    ) -> dict:
        sources, references = self._load_asset()

        predictions = predict_fn(sources)

        score = compute_sari(sources, predictions, references)

        result = {
            "asset_sari": score,
            "asset_split": self.split,
            "asset_max_samples": None if self.max_examples <= 0 else self.max_examples,
        }

        write_json(result, output_dir / "asset_sari_score.json")

        rows = [
            {
                "source": source,
                "prediction": prediction,
                "references": refs,
            }
            for source, prediction, refs in zip(sources, predictions, references, strict=True)
        ]

        write_json(rows, output_dir / "asset_sari_predictions.json")

        return result

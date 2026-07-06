from typing import Any

from datasets import load_dataset

from configuration.config import SEED
from data.dataset_loader import Pair
from prompts import simplify_prompt


class WikiSmallLoader:
    name = "wikismall"

    def __init__(
        self,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
        seed: int = SEED,
    ):
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.seed = seed

    def _to_pairs(self, split: Any, add_prompt: bool = True) -> list[Pair]:
        return [
            (
                simplify_prompt(row["text"]) if add_prompt else row["text"],
                row["text"],
            )
            for row in split
        ]

    def load_pairs(self, add_prompt: bool = True) -> tuple[list[Pair], list[Pair], list[Pair]]:
        dataset = load_dataset("acloudfan/wikismall")

        train_split = dataset["train"].shuffle(seed=self.seed)
        valid_split = dataset["validation"]
        test_split = dataset["test"] if "test" in dataset else dataset["validation"]

        if self.max_train_samples is not None:
            train_split = train_split.select(range(min(self.max_train_samples, len(train_split))))

        if self.max_eval_samples is not None:
            valid_split = valid_split.select(range(min(self.max_eval_samples, len(valid_split))))

            test_split = test_split.select(range(min(self.max_eval_samples, len(test_split))))

        train_pairs = self._to_pairs(train_split, add_prompt)
        valid_pairs = self._to_pairs(valid_split, add_prompt)
        test_pairs = self._to_pairs(test_split, add_prompt)

        return train_pairs, valid_pairs, test_pairs

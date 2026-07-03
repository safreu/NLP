from pathlib import Path
from random import Random

from data.dataset_loader import Pair
from prompts import simplify_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "wikismall"
FILE_PREFIX = "PWKP_108016.tag.80.aner.ori"
SPLITS = ("train", "valid", "test")


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.strip() for line in handle]


def load_raw_wikismall_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[Pair]:
    if split not in SPLITS:
        expected = ", ".join(SPLITS)
        raise ValueError(f"Unknown WikiSmall split {split!r}. Expected one of: {expected}.")

    source_path = data_dir / f"{FILE_PREFIX}.{split}.src"
    target_path = data_dir / f"{FILE_PREFIX}.{split}.dst"
    missing = [path for path in (source_path, target_path) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing WikiSmall data file(s): {joined}")

    sources = _read_lines(source_path)
    targets = _read_lines(target_path)
    if len(sources) != len(targets):
        raise ValueError(
            f"Mismatched WikiSmall/{split} line counts: "
            f"{source_path} has {len(sources)}, {target_path} has {len(targets)}."
        )

    pairs = [(source, target) for source, target in zip(sources, targets, strict=True)]
    empty_rows = [
        index
        for index, (source, target) in enumerate(pairs, start=1)
        if not source or not target
    ]
    if empty_rows:
        preview = ", ".join(str(index) for index in empty_rows[:5])
        raise ValueError(f"WikiSmall/{split} contains empty source or target rows at: {preview}")

    return pairs


def load_raw_wikismall_pairs(data_dir: Path = DEFAULT_DATA_DIR) -> list[Pair]:
    pairs: list[Pair] = []
    for split in SPLITS:
        pairs.extend(load_raw_wikismall_split(split, data_dir))
    return pairs


class WikiSmallLoader:
    name = "wikismall"

    def __init__(
        self,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
        seed: int = 42,
        data_dir: Path = DEFAULT_DATA_DIR,
    ):
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.seed = seed
        self.data_dir = data_dir

    def _to_pairs(self, split: list[Pair]) -> list[Pair]:
        return [(simplify_prompt(source), target) for source, target in split]

    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        train_split = load_raw_wikismall_split("train", self.data_dir)
        Random(self.seed).shuffle(train_split)
        valid_split = load_raw_wikismall_split("valid", self.data_dir)
        test_split = load_raw_wikismall_split("test", self.data_dir)

        if self.max_train_samples is not None:
            train_split = train_split[: self.max_train_samples]

        if self.max_eval_samples is not None:
            valid_split = valid_split[: self.max_eval_samples]
            test_split = test_split[: self.max_eval_samples]

        return (
            self._to_pairs(train_split),
            self._to_pairs(valid_split),
            self._to_pairs(test_split),
        )

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from configuration.config import MIN_LENGTH_RATIO, SIMILARITY_THRESHOLD
from evaluation.datasetStats import DatasetStats
from preprocessing.cleaner import clean_text
from preprocessing.filter import length_ratio, text_similarity
from prompts import simplify_prompt


@dataclass
class OneStopEnglishEntry:
    source: str
    target: str
    level: str
    source_file: str


@dataclass
class OneStopEnglish:
    entries: list[OneStopEnglishEntry]
    stats: DatasetStats = field(default_factory=DatasetStats)

    @staticmethod
    def _load_pair_file(file: Path, level: str) -> list[OneStopEnglishEntry]:
        text = file.read_text(encoding="utf-8", errors="replace")

        blocks = text.split("*******")

        pairs: list[OneStopEnglishEntry] = []

        for block in blocks:
            lines = [clean_text(line) for line in block.splitlines() if clean_text(line)]

            if len(lines) < 2:
                continue

            if file.name == "ELE-INT.txt":
                source = lines[1]
                target = lines[0]
            else:
                source = lines[0]
                target = lines[1]

            if not source or not target:
                continue

            pairs.append(
                OneStopEnglishEntry(
                    source=source, target=target, level=level, source_file=file.name
                )
            )

        return pairs

    @classmethod
    def load_from_disk(cls, path: str = "data/OneStopEnglishCorpus/Sentence-Aligned") -> Self:
        folder = Path(path)
        stats = DatasetStats()

        cache_file = folder.with_suffix(".pkl")

        """
        if cache_file.exists():
            with open(cache_file, "rb") as cached_file:
                entries = pickle.load(cached_file)

            return cls(entries)
        """

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exists {folder}")

        files = {
            "ADV-ELE.txt": "elementary",
            "ADV-INT.txt": "intermediate",
            "ELE-INT.txt": "elementary",
        }

        entries: list[OneStopEnglishEntry] = []

        for filename, level in files.items():
            file = folder / filename

            if not file.exists():
                raise FileNotFoundError(f"No {filename} in {folder}")

            loaded = cls._load_pair_file(file, level)

            stats.add_loaded(level, len(loaded))

            entries.extend(loaded)

        with open(cache_file, "wb") as cached_file:
            pickle.dump(entries, cached_file)

        return cls(entries, stats)

    def as_training_pairs(self, add_prompt: bool = True) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for entry in self.entries:
            raw_source = entry.source
            target = entry.target

            similarity = text_similarity(raw_source, target)
            ratio_score = length_ratio(raw_source, target)

            self.stats.similarity_scores.append(similarity)
            self.stats.length_ratios.append(ratio_score)

            if similarity > SIMILARITY_THRESHOLD:
                self.stats.skipped_similar += 1
                continue

            if ratio_score < MIN_LENGTH_RATIO:
                self.stats.skipped_length_ratio += 1
                continue

            source = simplify_prompt(raw_source) if add_prompt else raw_source

            training_pair = (source, target)

            if training_pair in seen:
                self.stats.skipped_duplicate += 1
                continue

            self.stats.add_kept(entry.level)
            seen.add(training_pair)
            pairs.append(training_pair)

        return pairs

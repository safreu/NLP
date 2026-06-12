import csv
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from cryptography.fernet import Fernet
from dotenv import load_dotenv

from config import MIN_LENGTH_RATIO, SIMILARITY_THRESHOLD
from evaluation.datasetStats import DatasetStats
from preprocessing.cleaner import clean_text, remove_prompt
from preprocessing.filter import length_ratio, text_similarity
from prompts import simplify_prompt


def _get_cache_cipher() -> Fernet:
    key = os.getenv("NEWSELA_CACHE_KEY")
    if not key:
        raise ValueError("Environment variable NEWSELA_CACHE_KEY is not set")
    return Fernet(key)


def _save_encrypted_pickle(obj: object, path: Path) -> None:
    cipher = _get_cache_cipher()
    raw = pickle.dumps(obj)
    encrypted = cipher.encrypt(raw)
    path.write_bytes(encrypted)


def _load_encrypted_pickle(path: Path) -> object:
    cipher = _get_cache_cipher()
    encrypted = path.read_bytes()
    raw = cipher.decrypt(encrypted)
    return pickle.loads(raw)


@dataclass
class NewselaTarget:
    text: str
    file_path: str
    grade_level: float | None = None
    version: int | None = None


@dataclass
class NewselaEntry:
    slug: str
    source: str
    targets: list[NewselaTarget]
    file_path: str
    grade_level: float | None = None


@dataclass
class NewselaCorpus:
    entries: list[NewselaEntry]
    stats: DatasetStats = field(default_factory=DatasetStats)

    @classmethod
    def load_from_disk(
        cls, path: str = "data/newsela/newsela_article_corpus_2016-01-29/articles_metadata.csv"
    ) -> Self:

        load_dotenv()

        file = Path(path)
        stats = DatasetStats()

        cache_file = file.with_suffix(".pkl.enc")

        if cache_file.exists():
            entries = _load_encrypted_pickle(cache_file)
            return cls(entries)

        if not file.exists():
            raise FileNotFoundError(f"File does not exist {file}")

        grouped: dict[str, list[dict]] = defaultdict(list)

        with open(file, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                grouped[row["slug"]].append(row)

        entries: list[NewselaEntry] = []

        for slug, rows in grouped.items():
            rows.sort(key=lambda r: int(r["version"]))

            source_row = rows[0]

            source = clean_text(source_row["text"])

            targets: list[NewselaTarget] = []

            for row in rows[1:]:
                target_text = clean_text(row["text"])

                if not target_text:
                    continue

                grade_level = float(row["grade_level"]) if row["grade_level"] else None

                targets.append(
                    NewselaTarget(
                        text=target_text,
                        grade_level=grade_level,
                        version=int(row["version"]),
                        file_path=str(row["filename"]),
                    )
                )

                stats_key = f"grade_{grade_level:g}" if grade_level is not None else "grade_unknown"

                stats.add_loaded(stats_key, 1)

            if not targets:
                continue

            entries.append(
                NewselaEntry(
                    slug=slug,
                    source=source,
                    grade_level=float(source_row["grade_level"])
                    if source_row["grade_level"]
                    else None,
                    targets=targets,
                    file_path=str(source_row["filename"]),
                )
            )

        _save_encrypted_pickle(entries, cache_file)

        return cls(entries, stats)

    def as_training_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for entry in self.entries:
            for target_entry in entry.targets:
                source = simplify_prompt(entry.source)

                target = target_entry.text
                cleaned_source = remove_prompt(source)

                similarity = text_similarity(cleaned_source, target)
                ratio_score = length_ratio(cleaned_source, target)

                self.stats.similarity_scores.append(similarity)
                self.stats.length_ratios.append(ratio_score)

                if similarity > SIMILARITY_THRESHOLD:
                    self.stats.skipped_similar += 1
                    continue

                if ratio_score < MIN_LENGTH_RATIO:
                    self.stats.skipped_length_ratio += 1
                    continue

                training_pair = (source, target)

                if training_pair in seen:
                    self.stats.skipped_duplicate += 1
                    continue

                stats_key = (
                    f"grade_{target_entry.grade_level}"
                    if target_entry.grade_level is not None
                    else "grade_unknown"
                )

                self.stats.add_kept(stats_key)
                seen.add(training_pair)
                pairs.append(training_pair)

        return pairs

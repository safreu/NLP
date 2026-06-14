import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from cryptography.fernet import Fernet
from dotenv import load_dotenv

from config import MIN_LENGTH_RATIO, SIMILARITY_THRESHOLD
from evaluation.datasetStats import DatasetStats
from preprocessing.cleaner import clean_text, detokenize_text, remove_prompt
from preprocessing.filter import length_ratio, text_similarity
from prompts import simplify_prompt


def _get_cache_cipher() -> Fernet:
    key = os.getenv("NEWSELA_CACHE_KEY")
    if not key:
        raise ValueError("Environment variable NEWSELA_CACHE_KEY is not set")
    return Fernet(key.encode())


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
class NewselaEntry:
    source: str
    target: str
    source_level: int | None = None
    target_level: int | None = None
    doc_id: str | None = None


DEFAULT_NEWSELA_PATH = Path(
    "data/newsela/newsela_article_corpus_2016-01-29"
    "/newsela_data_share/newsela_data_share-20150302"
    "/newsela_articles_20150302.aligned.sents.txt"
)


@dataclass
class NewselaCorpus:
    entries: list[NewselaEntry]
    stats: DatasetStats = field(default_factory=DatasetStats)

    @classmethod
    def load_from_disk(
        cls,
        path: str = DEFAULT_NEWSELA_PATH,
    ) -> Self:

        load_dotenv()

        file = Path(path)
        stats = DatasetStats()

        cache_dir = Path("cache/newsela")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{file.stem}.pkl.enc"

        if cache_file.exists():
            entries = _load_encrypted_pickle(cache_file)
            return cls(entries, stats)

        if not file.exists():
            raise FileNotFoundError(f"File does not exist {file}")

        entries: list[NewselaEntry] = []

        with open(file, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) != 5:
                    print(f"Skipping malformed line: {line}")
                    continue

                doc_id: str = clean_text(parts[0])
                source_level = clean_text(parts[1])
                target_level = clean_text(parts[2])
                source: str = clean_text(parts[3])
                target: str = clean_text(parts[4])

                try:
                    source_level = int(source_level.removeprefix("V"))
                    target_level = int(target_level.removeprefix("V"))
                except ValueError:
                    continue

                if target_level <= source_level:
                    print(f"{target_level} <= {source_level} for {line}")
                    continue

                entry = NewselaEntry(
                    doc_id=doc_id,
                    source_level=source_level,
                    target_level=target_level,
                    source=detokenize_text(source),
                    target=detokenize_text(target),
                )

                entries.append(entry)
                print(entry)
                stats.add_loaded(
                    f"{source_level}_to_{target_level}",
                    1,
                )

        _save_encrypted_pickle(entries, cache_file)

        return cls(entries, stats)

    def as_training_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for entry in self.entries:
            source = simplify_prompt(entry.source)
            cleaned_source = remove_prompt(source)

            target = entry.target

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
                f"{entry.source_level}_to_{entry.target_level}"
                if entry.source_level is not None and entry.target_level is not None
                else "level_unknown"
            )

            self.stats.add_kept(stats_key)
            seen.add(training_pair)
            pairs.append(training_pair)

        return pairs

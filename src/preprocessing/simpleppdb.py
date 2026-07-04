"""SimplePPDB++ download and lexical replacement extraction."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from preprocessing.classical_replacements import ReplacementDictionary
from preprocessing.classical_text import is_word, normalize_token

SIMPLEPPDB_REPOSITORY_URL = "https://github.com/mounicam/lexical_simplification"
SIMPLEPPDB_DATA_URL = (
    "https://media.githubusercontent.com/media/mounicam/lexical_simplification/"
    "master/SimplePPDBpp/simpleppdbpp_xl.tsv.gz"
)


@dataclass(frozen=True)
class SimplePPDBMetadata:
    repository_url: str
    data_url: str
    source_path: str
    min_score: float
    max_candidates_per_source: int
    lexical_only: bool
    source_vocabulary_size: int | None
    rule_limit: int | None
    rules_seen: int
    usable_rules: int
    replacement_sources: int
    replacement_candidates: int


def download_simpleppdb(path: Path, url: str = SIMPLEPPDB_DATA_URL) -> None:
    """Download the SimplePPDB++ gzip file if it is not already present."""
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(path)


def is_lexical_phrase(phrase: str) -> bool:
    return " " not in phrase.strip() and is_word(phrase.strip())


def iter_simpleppdb_rows(path: Path) -> Iterable[tuple[str, str, float, float]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                complexity_score = float(fields[2])
                ppdb_score = float(fields[3])
            except ValueError:
                continue
            yield fields[0], fields[1], complexity_score, ppdb_score


def score_to_count(score: float) -> int:
    return max(1, round(abs(score) * 1000))


def prune_candidates(
    dictionary: ReplacementDictionary,
    max_candidates_per_source: int,
) -> ReplacementDictionary:
    if max_candidates_per_source <= 0:
        return dictionary

    pruned = ReplacementDictionary(
        lowercase=dictionary.lowercase,
        min_count=dictionary.min_count,
    )
    for source, candidates in dictionary.counts.items():
        for target, count in candidates.most_common(max_candidates_per_source):
            pruned.add_count(source, target, int(count))
    return pruned


def collect_simpleppdb_replacements(
    path: Path,
    *,
    source_vocabulary: set[str] | None = None,
    lowercase: bool = True,
    min_count: int = 1,
    min_score: float = 0.0,
    max_candidates_per_source: int = 5,
    lexical_only: bool = True,
    rule_limit: int | None = None,
) -> tuple[ReplacementDictionary, SimplePPDBMetadata]:
    """Build a ReplacementDictionary from SimplePPDB++ readability-ranked rules.

    A positive SimplePPDB++ complexity score means phrase1 is more complex than
    phrase2; a negative score means the reverse. The absolute score is converted
    to an integer support weight so existing replacement-count features continue
    to work with the classical classifier.
    """
    dictionary = ReplacementDictionary(lowercase=lowercase, min_count=min_count)
    rules_seen = 0
    usable_rules = 0

    for phrase1, phrase2, complexity_score, _ in iter_simpleppdb_rows(path):
        rules_seen += 1
        if rule_limit is not None and rules_seen > rule_limit:
            break
        if abs(complexity_score) < min_score or complexity_score == 0:
            continue
        if lexical_only and not (is_lexical_phrase(phrase1) and is_lexical_phrase(phrase2)):
            continue

        complex_phrase, simple_phrase = (
            (phrase1, phrase2) if complexity_score > 0 else (phrase2, phrase1)
        )
        source = normalize_token(complex_phrase, lowercase)
        target = normalize_token(simple_phrase, lowercase)
        if source_vocabulary is not None and source not in source_vocabulary:
            continue
        if not source or not target or source == target:
            continue

        dictionary.add_count(source, target, score_to_count(complexity_score))
        usable_rules += 1

    dictionary = prune_candidates(dictionary, max_candidates_per_source)
    replacement_candidates = sum(len(candidates) for candidates in dictionary.counts.values())
    metadata = SimplePPDBMetadata(
        repository_url=SIMPLEPPDB_REPOSITORY_URL,
        data_url=SIMPLEPPDB_DATA_URL,
        source_path=str(path),
        min_score=min_score,
        max_candidates_per_source=max_candidates_per_source,
        lexical_only=lexical_only,
        source_vocabulary_size=len(source_vocabulary) if source_vocabulary is not None else None,
        rule_limit=rule_limit,
        rules_seen=rules_seen,
        usable_rules=usable_rules,
        replacement_sources=len(dictionary.counts),
        replacement_candidates=replacement_candidates,
    )
    return dictionary, metadata

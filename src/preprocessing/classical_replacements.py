"""Replacement dictionary construction and persistence."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from preprocessing.classical_text import normalize_token
from storage.classical_io import atomic_write_json


@dataclass
class ReplacementDictionary:
    """Counts and selects lexical simplification replacements."""

    counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    lowercase: bool = True
    min_count: int = 1

    def add(self, complex_word: str, simple_word: str) -> None:
        source = normalize_token(complex_word, self.lowercase)
        target = normalize_token(simple_word, self.lowercase)
        if source and target and source != target:
            self.counts[source][target] += 1

    def best_replacement(self, complex_word: str) -> str | None:
        source = normalize_token(complex_word, self.lowercase)
        candidates = self.counts.get(source)
        if not candidates:
            return None
        replacement, count = candidates.most_common(1)[0]
        if count < self.min_count:
            return None
        return replacement

    def has_replacement(self, complex_word: str) -> bool:
        return self.best_replacement(complex_word) is not None

    def replacement_count(self, complex_word: str, simple_word: str | None = None) -> int:
        source = normalize_token(complex_word, self.lowercase)
        candidates = self.counts.get(source)
        if not candidates:
            return 0
        if simple_word is None:
            return sum(candidates.values())
        return candidates[normalize_token(simple_word, self.lowercase)]

    def replacement_frequency(self, complex_word: str, simple_word: str | None = None) -> float:
        source = normalize_token(complex_word, self.lowercase)
        candidates = self.counts.get(source)
        if not candidates:
            return 0.0
        total = sum(candidates.values())
        if not total:
            return 0.0
        if simple_word is None:
            return candidates.most_common(1)[0][1] / total
        return candidates[normalize_token(simple_word, self.lowercase)] / total

    def to_dict(self) -> dict[str, object]:
        replacements: dict[str, object] = {}
        for source, counter in sorted(self.counts.items()):
            total = sum(counter.values())
            best = counter.most_common(1)[0][0] if counter else None
            replacements[source] = {
                "total_count": total,
                "best_replacement": best,
                "candidates": [
                    {"replacement": target, "count": count, "frequency": count / total}
                    for target, count in counter.most_common()
                ],
            }
        return {
            "lowercase": self.lowercase,
            "min_count": self.min_count,
            "replacements": replacements,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ReplacementDictionary:
        raw_min_count = data.get("min_count", 1)
        if not isinstance(raw_min_count, int | str):
            raw_min_count = 1
        dictionary = cls(
            lowercase=bool(data.get("lowercase", True)),
            min_count=int(raw_min_count),
        )
        replacements = data.get("replacements", {})
        if not isinstance(replacements, dict):
            return dictionary
        for source, payload in replacements.items():
            if not isinstance(payload, dict):
                continue
            for candidate in payload.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                target = str(candidate["replacement"])
                count = int(candidate["count"])
                dictionary.counts[str(source)][target] = count
        return dictionary

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> ReplacementDictionary:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

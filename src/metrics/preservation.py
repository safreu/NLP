"""Corpus-level number and named-entity preservation metrics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

_NER_PIPELINE: Any | None = None
NUMBER_PATTERN = re.compile(r"(?<!\w)[-+]?(?:\d{1,3}(?:,\s*\d{3})+|\d+)(?:\s*\.\s*\d+)?\s*%?(?!\w)")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _overlap(left: Counter[Any], right: Counter[Any]) -> int:
    return sum(min(count, right[item]) for item, count in left.items())


def _normalize_number(value: str) -> str:
    compact = re.sub(r"\s+", "", value).replace(",", "")
    is_percent = compact.endswith("%")
    number = compact.removesuffix("%")
    if "." in number:
        with suppress(ValueError):
            number = format(float(number), "g")
    return f"{number}%" if is_percent else number


def _numbers(text: str) -> Counter[str]:
    return Counter(_normalize_number(match.group()) for match in NUMBER_PATTERN.finditer(text))


def compute_number_preservation(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[str],
) -> dict[str, float | int]:
    if not (len(sources) == len(candidates) == len(references)):
        raise ValueError("Source, candidate, and reference lengths must match.")

    total = candidate_preserved = reference_preserved = 0
    for source, candidate, reference in zip(sources, candidates, references, strict=True):
        source_numbers = _numbers(source)
        total += sum(source_numbers.values())
        candidate_preserved += _overlap(source_numbers, _numbers(candidate))
        reference_preserved += _overlap(source_numbers, _numbers(reference))

    return {
        "source_number_count": total,
        "candidate_preserved_count": candidate_preserved,
        "candidate_lost_count": total - candidate_preserved,
        "candidate_preservation_rate": _rate(candidate_preserved, total),
        "reference_preserved_count": reference_preserved,
        "reference_preservation_rate": _rate(reference_preserved, total),
    }


def _ner_pipeline() -> Any:
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        import spacy

        try:
            _NER_PIPELINE = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "Named-entity preservation requires en_core_web_sm. "
                "Install the project dependencies with `uv sync`."
            ) from exc
    return _NER_PIPELINE


def _entity_counts(document: Any) -> Counter[tuple[str, str]]:
    return Counter(
        (re.sub(r"\s+", " ", entity.text.strip().lower()).strip(".,;:!?\"'`()[]{}"), entity.label_)
        for entity in document.ents
        if entity.text.strip()
    )


def compute_entity_preservation(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[str],
) -> dict[str, object]:
    """Compare exact normalized entity text and spaCy entity type."""
    if not (len(sources) == len(candidates) == len(references)):
        raise ValueError("Source, candidate, and reference lengths must match.")

    nlp = _ner_pipeline()
    total = candidate_preserved = reference_preserved = 0
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"source_count": 0, "candidate_preserved_count": 0}
    )
    document_stream = zip(
        nlp.pipe(sources),
        nlp.pipe(candidates),
        nlp.pipe(references),
        strict=True,
    )
    for source_doc, candidate_doc, reference_doc in document_stream:
        source_entities = _entity_counts(source_doc)
        candidate_entities = _entity_counts(candidate_doc)
        reference_entities = _entity_counts(reference_doc)
        total += sum(source_entities.values())
        candidate_preserved += _overlap(source_entities, candidate_entities)
        reference_preserved += _overlap(source_entities, reference_entities)
        for entity, count in source_entities.items():
            entity_type = entity[1]
            by_type[entity_type]["source_count"] += count
            by_type[entity_type]["candidate_preserved_count"] += min(
                count, candidate_entities[entity]
            )

    type_metrics = {
        entity_type: {
            **counts,
            "candidate_preservation_rate": _rate(
                counts["candidate_preserved_count"], counts["source_count"]
            ),
        }
        for entity_type, counts in sorted(by_type.items())
    }
    return {
        "matching_rule": "normalized_text_and_entity_type",
        "source_entity_count": total,
        "candidate_preserved_count": candidate_preserved,
        "candidate_lost_count": total - candidate_preserved,
        "candidate_preservation_rate": _rate(candidate_preserved, total),
        "reference_preserved_count": reference_preserved,
        "reference_preservation_rate": _rate(reference_preserved, total),
        "by_type": type_metrics,
    }


def compute_preservation_metrics(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[str],
) -> dict[str, object]:
    return {
        "entity_preservation": compute_entity_preservation(sources, candidates, references),
        "number_preservation": compute_number_preservation(sources, candidates, references),
    }

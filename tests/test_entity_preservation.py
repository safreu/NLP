from collections import Counter

from evaluation.entity_preservation import (
    counter_lost_items,
    counter_preserved_items,
    detect_columns,
    entity_key,
    normalize_entity_text,
    overlap_count,
)


def test_normalize_entity_text_removes_case_spacing_and_punctuation() -> None:
    assert normalize_entity_text("  New   York, ") == "new york"


def test_entity_key_uses_normalized_text_and_type() -> None:
    entity = {"normalized_text": "ulm", "entity_type": "GPE"}

    assert entity_key(entity) == ("ulm", "GPE")


def test_counter_helpers_track_preserved_and_lost_duplicates() -> None:
    original = Counter({("ulm", "GPE"): 2, ("monday", "DATE"): 1})
    output = Counter({("ulm", "GPE"): 1})

    assert overlap_count(original, output) == 1
    assert counter_preserved_items(original, output) == [("ulm", "GPE")]
    assert counter_lost_items(original, output) == [("monday", "DATE"), ("ulm", "GPE")]


def test_detect_columns_accepts_standard_candidate_alias() -> None:
    assert detect_columns(["source", "candidate", "reference"]) == ("source", "candidate")

# mypy: ignore-errors
"""Feature Set A extraction for replacement candidates."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.neural_replacement_filter.rules import is_connector

SPACY_MODEL_NAME = "en_core_web_sm"


@dataclass(frozen=True)
class CandidateFeatures:
    """Feature Set A for one original/replacement pair in context."""

    model_confidence: float
    model_source: str
    pos_tag: str
    ner_tag: str
    is_proper_noun: bool
    is_number: bool
    is_connector: bool
    original_length: int
    replacement_length: int
    length_difference: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_confidence": self.model_confidence,
            "model_source": self.model_source,
            "pos_tag": self.pos_tag,
            "ner_tag": self.ner_tag,
            "is_proper_noun": self.is_proper_noun,
            "is_number": self.is_number,
            "is_connector": self.is_connector,
            "original_length": self.original_length,
            "replacement_length": self.replacement_length,
            "length_difference": self.length_difference,
        }


def load_spacy_model():
    """Load spaCy and emit a helpful setup error when the model is missing."""

    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is not installed. Install project dependencies first.") from exc

    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL_NAME}' is missing. Run:\n"
            f"python -m spacy download {SPACY_MODEL_NAME}"
        ) from exc


def extract_feature_set_a(
    *,
    token,
    replacement_word: str,
    model_confidence: float,
    model_source: str,
) -> CandidateFeatures:
    """Extract the requested Feature Set A columns from a spaCy token."""

    original_word = token.text
    pos_tag = token.pos_ or ""
    ner_tag = token.ent_type_ or ""
    original_length = len(original_word)
    replacement_length = len(replacement_word)

    return CandidateFeatures(
        model_confidence=float(model_confidence),
        model_source=model_source,
        pos_tag=pos_tag,
        ner_tag=ner_tag,
        is_proper_noun=pos_tag == "PROPN",
        is_number=bool(token.like_num or pos_tag == "NUM"),
        is_connector=is_connector(original_word),
        original_length=original_length,
        replacement_length=replacement_length,
        length_difference=replacement_length - original_length,
    )

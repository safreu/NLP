# mypy: ignore-errors
"""Hard safety rules for replacement candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass

PROTECTED_CONNECTORS = {
    "although",
    "however",
    "therefore",
    "because",
    "since",
    "while",
    "before",
    "after",
    "during",
    "despite",
    "unless",
    "whereas",
    "if",
    "then",
    "but",
    "so",
}

PROTECTED_POS_TAGS = {
    "PRON",
    "DET",
    "ADP",
    "AUX",
    "CCONJ",
    "SCONJ",
    "PART",
}

PROTECTED_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "by",
    "with",
    "from",
    "for",
    "on",
    "at",
    "as",
    "and",
    "or",
    "but",
    "so",
    "if",
    "then",
    "that",
    "which",
    "who",
    "he",
    "she",
    "it",
    "they",
    "them",
    "his",
    "her",
    "their",
    "our",
    "your",
    "we",
    "you",
}

BRACKET_ARTIFACTS = {"rrb", "lrb", "-rrb-", "-lrb-"}
MISSPELLING_ARTIFACTS = {"askr", "propection", "violece"}
ISOLATED_BAD_LETTERS = {"s", "g", "x", "r"}
MONTH_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
ORDINAL_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
}
QUANTITY_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "hundred",
    "thousand",
    "million",
    "billion",
}
SENSITIVE_CATEGORY_WORDS = {
    "woman",
    "women",
    "man",
    "men",
    "male",
    "female",
    "person",
    "people",
    "child",
    "children",
    "mother",
    "father",
    "son",
    "daughter",
    "president",
    "mayor",
    "king",
    "queen",
    "prince",
    "princess",
    "country",
    "nation",
    "city",
    "village",
    "family",
    "group",
}
BAD_REPLACEMENT_BLACKLIST = {
    "although",
    "north",
    "pandas",
    "storm",
    "pork",
    "rick",
    "s",
    "g",
    "rrb",
    "lrb",
    "askr",
    "propection",
    "violece",
    "songs",
    "movie",
    "first",
    "two",
    "thing",
    "stuff",
    "been",
}
UNSAFE_CONTENT_REPLACEMENT_PAIRS = {
    ("convection", "strength"),
    ("content", "images"),
    ("computer", "video"),
}
VAGUE_REPLACEMENTS = {
    "thing",
    "things",
    "stuff",
    "people",
    "make",
    "made",
    "do",
    "does",
    "did",
    "good",
    "bad",
}
CONTENT_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
COMPATIBLE_CONTENT_POS = {
    ("NOUN", "NOUN"),
    ("PROPN", "PROPN"),
    ("PROPN", "NOUN"),
    ("VERB", "VERB"),
    ("ADJ", "ADJ"),
    ("ADV", "ADV"),
}

PUNCTUATION_ONLY_RE = re.compile(r"^\W+$", re.UNICODE)
WORDISH_RE = re.compile(r"^[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*$")
CLEAN_ALPHA_RE = re.compile(r"[^A-Za-z]+")


@dataclass(frozen=True)
class HardRuleDecision:
    """Decision returned by the hard-rule layer."""

    blocked: bool
    reason: str


def is_connector(token: str) -> bool:
    return token.lower() in PROTECTED_CONNECTORS


def is_function_word_or_pronoun(original_word: str, replacement_word: str, pos_tag: str) -> bool:
    return (
        pos_tag in PROTECTED_POS_TAGS
        or original_word.lower() in PROTECTED_FUNCTION_WORDS
        or replacement_word.lower() in PROTECTED_FUNCTION_WORDS
    )


def clean_alpha(token: str) -> str:
    return CLEAN_ALPHA_RE.sub("", token).lower()


def same_surface(left: str, right: str) -> bool:
    return clean_alpha(left) == clean_alpha(right)


def is_malformed(token: str, replacement: str | None = None) -> bool:
    if not token or token.isspace():
        return True
    if PUNCTUATION_ONLY_RE.match(token):
        return True
    if not WORDISH_RE.match(token):
        return True
    return replacement is not None and (not replacement or replacement.isspace())


def is_malformed_replacement(original_word: str, replacement_word: str) -> bool:
    cleaned = clean_alpha(replacement_word)
    replacement_lower = replacement_word.lower()
    original_cleaned = clean_alpha(original_word)
    if replacement_lower in BRACKET_ARTIFACTS | MISSPELLING_ARTIFACTS:
        return True
    if replacement_lower in ISOLATED_BAD_LETTERS:
        return True
    if PUNCTUATION_ONLY_RE.match(replacement_word):
        return True
    if len(cleaned) == 1 and cleaned not in {"a", "i"}:
        return True
    if not cleaned and not original_word.isnumeric():
        return True
    normalized_replacement = replacement_lower.replace("-", "").replace("'", "")
    return cleaned != normalized_replacement and not original_cleaned.isnumeric()


def is_number_or_date_change(original_word: str, replacement_word: str, is_number: bool) -> bool:
    original = clean_alpha(original_word)
    replacement = clean_alpha(replacement_word)
    if is_number and not same_surface(original_word, replacement_word):
        return True
    if original_word.isnumeric() and original_word != replacement_word:
        return True
    if original in MONTH_WORDS and replacement != original:
        return True
    if replacement in MONTH_WORDS and replacement != original:
        return True
    if original in ORDINAL_WORDS and replacement != original:
        return True
    if replacement in ORDINAL_WORDS and replacement != original:
        return True
    if original in QUANTITY_WORDS and replacement != original:
        return True
    return replacement in QUANTITY_WORDS and replacement != original


def is_sensitive_category_change(original_word: str, replacement_word: str) -> bool:
    original = clean_alpha(original_word)
    replacement = clean_alpha(replacement_word)
    return original != replacement and (
        original in SENSITIVE_CATEGORY_WORDS or replacement in SENSITIVE_CATEGORY_WORDS
    )


def is_blacklisted_replacement(original_word: str, replacement_word: str) -> bool:
    original = clean_alpha(original_word)
    replacement = clean_alpha(replacement_word)
    return (
        (original, replacement) in UNSAFE_CONTENT_REPLACEMENT_PAIRS
        or replacement in BAD_REPLACEMENT_BLACKLIST
    ) and not same_surface(original_word, replacement_word)


def is_pos_incompatible(pos_tag: str, replacement_pos_tag: str) -> bool:
    if not pos_tag or not replacement_pos_tag:
        return False
    if replacement_pos_tag in PROTECTED_POS_TAGS and pos_tag in CONTENT_POS_TAGS:
        return True
    if pos_tag in CONTENT_POS_TAGS and replacement_pos_tag in CONTENT_POS_TAGS:
        return (pos_tag, replacement_pos_tag) not in COMPATIBLE_CONTENT_POS
    return False


def is_vague_replacement(original_word: str, replacement_word: str, pos_tag: str) -> bool:
    replacement = clean_alpha(replacement_word)
    return (
        replacement in VAGUE_REPLACEMENTS
        and not same_surface(original_word, replacement_word)
        and pos_tag in CONTENT_POS_TAGS
    )


def apply_hard_rules(
    *,
    original_word: str,
    replacement_word: str,
    pos_tag: str,
    ner_tag: str,
    is_proper_noun: bool,
    is_number: bool,
    is_connector_token: bool,
    replacement_pos_tag: str = "",
) -> HardRuleDecision:
    """Return whether a candidate should be blocked before neural filtering."""

    reasons: list[str] = []
    is_lowercase_safe_entity = (
        original_word.islower()
        and not is_number
        and not is_connector_token
        and not is_proper_noun
        and pos_tag != "PROPN"
    )
    if is_malformed(original_word, replacement_word):
        reasons.append("malformed_or_punctuation")
    if is_function_word_or_pronoun(original_word, replacement_word, pos_tag):
        return HardRuleDecision(blocked=True, reason="function_word_or_pronoun")
    if (is_proper_noun or pos_tag == "PROPN" or ner_tag) and not same_surface(
        original_word, replacement_word
    ):
        return HardRuleDecision(blocked=True, reason="content_quality:proper_noun_replacement")
    if is_malformed_replacement(original_word, replacement_word):
        return HardRuleDecision(blocked=True, reason="content_quality:malformed_replacement")
    if is_number_or_date_change(original_word, replacement_word, is_number):
        return HardRuleDecision(blocked=True, reason="content_quality:number_or_date_change")
    if is_sensitive_category_change(original_word, replacement_word):
        return HardRuleDecision(blocked=True, reason="content_quality:sensitive_category_change")
    if is_blacklisted_replacement(original_word, replacement_word):
        return HardRuleDecision(blocked=True, reason="content_quality:blacklisted_replacement")
    if is_pos_incompatible(pos_tag, replacement_pos_tag):
        return HardRuleDecision(blocked=True, reason="content_quality:pos_incompatible")
    if is_vague_replacement(original_word, replacement_word, pos_tag):
        return HardRuleDecision(blocked=True, reason="content_quality:vague_replacement")
    if is_proper_noun or pos_tag == "PROPN":
        reasons.append("proper_noun")
    if ner_tag and not is_lowercase_safe_entity:
        reasons.append(f"named_entity:{ner_tag}")
    if is_number:
        reasons.append("number")
    if is_connector_token:
        reasons.append("protected_connector")

    return HardRuleDecision(blocked=bool(reasons), reason=";".join(reasons))

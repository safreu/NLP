"""Small text-processing helpers for the classical simplifier."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from typing import cast

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "''", "'s"}
NO_SPACE_AFTER = {"(", "[", "{", "``", "$", "#"}


def tokenize(text: str) -> list[str]:
    """Tokenize with NLTK when available, otherwise use a regex fallback."""
    try:
        from nltk.tokenize import word_tokenize

        return cast(list[str], word_tokenize(text))
    except Exception:
        return TOKEN_RE.findall(text)


def is_word(token: str) -> bool:
    return bool(WORD_RE.match(token))


def normalize_token(token: str, lowercase: bool = True) -> str:
    return token.lower() if lowercase else token


def detokenize(tokens: Iterable[str]) -> str:
    """Reconstruct a readable sentence from tokens."""
    output = ""
    previous = ""
    for token in tokens:
        if not output:
            output = token
        elif token in NO_SPACE_BEFORE or previous in NO_SPACE_AFTER:
            output += token
        else:
            output += " " + token
        previous = token
    return output


@lru_cache(maxsize=50000)
def syllable_count(word: str) -> int:
    """Approximate syllable count without external corpora."""
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    groups = re.findall(r"[aeiouy]+", cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def average_word_length(tokens: list[str]) -> float:
    words = [token for token in tokens if is_word(token)]
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def flesch_reading_ease_proxy(tokens: list[str]) -> float:
    """A single-sentence Flesch-style proxy used as a feature."""
    words = [token for token in tokens if is_word(token)]
    if not words:
        return 0.0
    syllables = sum(syllable_count(word) for word in words)
    return 206.835 - 1.015 * len(words) - 84.6 * (syllables / len(words))


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance implemented locally to avoid an extra dependency."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def sequence_similarity(left: str, right: str) -> float:
    from difflib import SequenceMatcher

    if not left and not right:
        return 1.0
    return SequenceMatcher(a=left.lower(), b=right.lower()).ratio()


def safe_pos_tags(tokens: list[str]) -> list[str]:
    """Return POS tags, falling back to simple tags if NLTK data is missing."""
    try:
        from nltk import pos_tag

        return [tag for _, tag in pos_tag(tokens)]
    except Exception:
        return ["WORD" if is_word(token) else "PUNCT" for token in tokens]


@lru_cache(maxsize=50000)
def lemmatize_token(token: str, pos_tag: str = "") -> str:
    """Lemmatize with WordNet when available, otherwise lowercase."""
    try:
        from nltk.stem import WordNetLemmatizer

        wordnet_pos = "n"
        if pos_tag.startswith("V"):
            wordnet_pos = "v"
        elif pos_tag.startswith("J"):
            wordnet_pos = "a"
        elif pos_tag.startswith("R"):
            wordnet_pos = "r"
        return cast(str, WordNetLemmatizer().lemmatize(token.lower(), wordnet_pos))
    except Exception:
        return token.lower()

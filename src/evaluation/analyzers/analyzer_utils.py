from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NUMBER_RE = re.compile(
    r"(?<!\w)[-+]?(?:\d{1,3}(?:,\s*\d{3})+|\d+)"
    r"(?:\s*\.\s*\d+)?\s*%?(?!\w)"
)


def tokens(text: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def words(text: Any) -> list[str]:
    return WORD_RE.findall(str(text))


def normalize_text(text: Any) -> str:
    return " ".join(tokens(text))


def token_f1(left: Any, right: Any) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)

    if not left_tokens and not right_tokens:
        return 1.0

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)

    return 2 * precision * recall / (precision + recall)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_number(number: str) -> str:
    number = number.strip()

    has_percent = number.replace(" ", "").endswith("%")
    core = number.rstrip("%") if has_percent else number

    core = re.sub(r"\s+", "", core.replace(",", ""))

    if "." in core:
        with suppress(ValueError):
            core = str(float(core)).rstrip("0").rstrip(".")

    return f"{core}%" if has_percent else core


def extract_numbers(text: Any) -> list[str]:
    return [normalize_number(match.group(0)) for match in NUMBER_RE.finditer(str(text))]


def preservation_rate(source_items: Iterable[str], output_items: Iterable[str]) -> float | None:
    source_counts = Counter(source_items)

    if not source_counts:
        return None

    output_counts = Counter(output_items)

    preserved = sum(min(count, output_counts[item]) for item, count in source_counts.items())

    return preserved / sum(source_counts.values())

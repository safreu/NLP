"""Build supervised examples from aligned simplification pairs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import cast

from data.dataset_loader import Pair
from preprocessing.classical_features import FeatureExtractor
from preprocessing.classical_replacements import ReplacementDictionary
from preprocessing.classical_text import is_word, normalize_token, tokenize
from preprocessing.cleaner import remove_prompt

KEEP = "KEEP"
SIMPLIFY = "SIMPLIFY"


def progress_iter[T](iterable: Iterable[T], **kwargs: object) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return cast(Iterable[T], tqdm(iterable, **kwargs))


@dataclass(frozen=True)
class RawTrainingExample:
    pair_id: int
    source_sentence: str
    token: str
    token_index: int
    label: str
    replacement: str | None = None


@dataclass(frozen=True)
class TrainingExample:
    token: str
    label: str
    features: dict[str, object]
    replacement: str | None = None


def to_classical_pairs(pairs: list[Pair]) -> list[Pair]:
    return [(remove_prompt(source), target) for source, target in pairs]


def build_token_frequencies(pairs: list[Pair], lowercase: bool = True) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for source, target in progress_iter(pairs, desc="Token frequencies", unit="pair"):
        for sentence in (source, target):
            for token in tokenize(sentence):
                if is_word(token):
                    frequencies[normalize_token(token, lowercase)] += 1
    return frequencies


def choose_replacements(source_tokens: list[str], target_tokens: list[str]) -> dict[int, str]:
    """Pick simple single-word candidates from a replaced span.

    Returns a map from source-token offset to replacement token.
    """
    source_word_positions = [
        (index, token) for index, token in enumerate(source_tokens) if is_word(token)
    ]
    target_words = [token for token in target_tokens if is_word(token)]
    if not source_word_positions or len(source_word_positions) != len(target_words):
        return {}
    replacements: dict[int, str] = {}
    for (source_index, source), target in zip(source_word_positions, target_words, strict=True):
        if source.lower() == target.lower():
            continue
        if len(target) > len(source) and len(target) > 6:
            continue
        replacements[source_index] = target
    return replacements


def choose_replacement(source_tokens: list[str], target_tokens: list[str]) -> str | None:
    """Backward-compatible helper for a single source-token replacement."""
    replacements = choose_replacements(source_tokens, target_tokens)
    if len(replacements) != 1:
        return None
    return next(iter(replacements.values()))


def collect_replacements(
    pairs: list[Pair],
    lowercase: bool = True,
    min_count: int = 1,
) -> ReplacementDictionary:
    dictionary = ReplacementDictionary(lowercase=lowercase, min_count=min_count)
    for source, target in progress_iter(pairs, desc="Replacement dictionary", unit="pair"):
        source_tokens = tokenize(source)
        target_tokens = tokenize(target)
        matcher = SequenceMatcher(
            a=[normalize_token(token, lowercase) for token in source_tokens],
            b=[normalize_token(token, lowercase) for token in target_tokens],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                replacements = choose_replacements(
                    source_tokens[i1:i2],
                    target_tokens[j1:j2],
                )
                for offset, replacement in replacements.items():
                    dictionary.add(source_tokens[i1 + offset], replacement)
    return dictionary


def build_training_examples(
    pairs: list[Pair],
    feature_extractor: FeatureExtractor,
    lowercase: bool = True,
) -> list[TrainingExample]:
    raw_examples = build_raw_training_examples(pairs, lowercase=lowercase)
    return extract_features_for_examples(raw_examples, feature_extractor)


def build_raw_training_examples(
    pairs: list[Pair],
    lowercase: bool = True,
) -> list[RawTrainingExample]:
    raw_examples: list[RawTrainingExample] = []
    for pair_id, (source, target) in progress_iter(
        enumerate(pairs),
        total=len(pairs),
        desc="Training examples",
        unit="pair",
    ):
        source_tokens = tokenize(source)
        target_tokens = tokenize(target)
        matcher = SequenceMatcher(
            a=[normalize_token(token, lowercase) for token in source_tokens],
            b=[normalize_token(token, lowercase) for token in target_tokens],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for index in range(i1, i2):
                    token = source_tokens[index]
                    if not is_word(token):
                        continue
                    raw_examples.append(
                        RawTrainingExample(
                            pair_id=pair_id,
                            source_sentence=source,
                            token=token,
                            token_index=index,
                            label=KEEP,
                        )
                    )
            elif tag == "replace":
                replacements = choose_replacements(
                    source_tokens[i1:i2],
                    target_tokens[j1:j2],
                )
                for index in range(i1, i2):
                    token = source_tokens[index]
                    if not is_word(token):
                        continue
                    replacement = replacements.get(index - i1)
                    label = SIMPLIFY if replacement is not None else KEEP
                    raw_examples.append(
                        RawTrainingExample(
                            pair_id=pair_id,
                            source_sentence=source,
                            token=token,
                            token_index=index,
                            label=label,
                            replacement=replacement,
                        )
                    )
            elif tag == "delete":
                for index in range(i1, i2):
                    token = source_tokens[index]
                    if not is_word(token):
                        continue
                    raw_examples.append(
                        RawTrainingExample(
                            pair_id=pair_id,
                            source_sentence=source,
                            token=token,
                            token_index=index,
                            label=KEEP,
                        )
                    )
    return raw_examples


def extract_features_for_examples(
    raw_examples: list[RawTrainingExample],
    feature_extractor: FeatureExtractor,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    context_cache: dict[tuple[int, str], dict[str, object]] = {}
    for raw in progress_iter(raw_examples, desc="Feature extraction", unit="example"):
        cache_key = (raw.pair_id, raw.source_sentence)
        if cache_key not in context_cache:
            context_cache[cache_key] = feature_extractor.sentence_context(raw.source_sentence)
        context = context_cache[cache_key]
        examples.append(
            TrainingExample(
                token=raw.token,
                label=raw.label,
                replacement=raw.replacement,
                features=feature_extractor.extract_for_token(
                    raw.token,
                    raw.token_index,
                    context,
                    raw.replacement,
                ),
            )
        )
    return examples

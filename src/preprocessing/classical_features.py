"""Feature extraction for classical lexical simplification."""

from __future__ import annotations

from collections import Counter

from preprocessing.classical_replacements import ReplacementDictionary
from preprocessing.classical_text import (
    average_word_length,
    edit_distance,
    flesch_reading_ease_proxy,
    is_word,
    lemmatize_token,
    normalize_token,
    safe_pos_tags,
    sequence_similarity,
    syllable_count,
    tokenize,
)


class FeatureExtractor:
    """Extract dictionary-style features consumed by DictVectorizer."""

    def __init__(
        self,
        token_frequencies: Counter[str] | None = None,
        replacement_dictionary: ReplacementDictionary | None = None,
        lowercase: bool = True,
    ) -> None:
        self.token_frequencies = token_frequencies or Counter()
        self.replacement_dictionary = replacement_dictionary or ReplacementDictionary(
            lowercase=lowercase
        )
        self.lowercase = lowercase
        self.total_tokens = max(sum(self.token_frequencies.values()), 1)

    def sentence_context(self, sentence: str) -> dict[str, object]:
        tokens = tokenize(sentence)
        return self.context_from_tokens(tokens)

    def context_from_tokens(self, tokens: list[str]) -> dict[str, object]:
        word_tokens = [token for token in tokens if is_word(token)]
        return {
            "tokens": tokens,
            "pos_tags": safe_pos_tags(tokens),
            "sentence_length": len(word_tokens),
            "token_count": len(tokens),
            "avg_word_length": average_word_length(tokens),
            "readability_proxy": flesch_reading_ease_proxy(tokens),
        }

    def extract_for_token(
        self,
        token: str,
        index: int,
        context: dict[str, object],
        candidate_replacement: str | None = None,
    ) -> dict[str, object]:
        tokens = context["tokens"]
        pos_tags = context["pos_tags"]
        assert isinstance(tokens, list)
        assert isinstance(pos_tags, list)

        normalized = normalize_token(token, self.lowercase)
        pos_tag = str(pos_tags[index]) if index < len(pos_tags) else "UNK"
        replacement = candidate_replacement or self.replacement_dictionary.best_replacement(token)
        frequency = self.token_frequencies[normalized]
        sentence_length_value = context["sentence_length"]
        token_count_value = context["token_count"]
        avg_word_length_value = context["avg_word_length"]
        readability_proxy_value = context["readability_proxy"]
        assert isinstance(sentence_length_value, int)
        assert isinstance(token_count_value, int)
        assert isinstance(avg_word_length_value, int | float)
        assert isinstance(readability_proxy_value, int | float)
        sentence_length = sentence_length_value
        token_count = token_count_value

        features: dict[str, object] = {
            "word": normalized,
            "lemma": lemmatize_token(token, pos_tag),
            "pos_tag": pos_tag,
            "is_word": is_word(token),
            "is_title": token.istitle(),
            "is_upper": token.isupper(),
            "word_length": len(token),
            "syllable_count": syllable_count(token),
            "token_frequency": frequency,
            "token_frequency_ratio": frequency / self.total_tokens,
            "token_position": index,
            "relative_position": index / max(token_count - 1, 1),
            "sentence_length": sentence_length,
            "token_count": token_count,
            "avg_word_length": float(avg_word_length_value),
            "readability_proxy": float(readability_proxy_value),
            "has_replacement": replacement is not None,
        }
        if replacement is not None:
            replacement_frequency = self.token_frequencies[
                normalize_token(replacement, self.lowercase)
            ]
            features.update(
                {
                    "replacement": normalize_token(replacement, self.lowercase),
                    "replacement_length": len(replacement),
                    "replacement_syllables": syllable_count(replacement),
                    "edit_distance": edit_distance(normalized, replacement),
                    "length_difference": len(token) - len(replacement),
                    "frequency_difference": replacement_frequency - frequency,
                    "similarity_score": sequence_similarity(token, replacement),
                    "replacement_count": self.replacement_dictionary.replacement_count(
                        token, replacement
                    ),
                    "replacement_frequency": self.replacement_dictionary.replacement_frequency(
                        token, replacement
                    ),
                }
            )
        else:
            features.update(
                {
                    "replacement": "<NONE>",
                    "replacement_length": 0,
                    "replacement_syllables": 0,
                    "edit_distance": 0,
                    "length_difference": 0,
                    "frequency_difference": 0,
                    "similarity_score": 0.0,
                    "replacement_count": 0,
                    "replacement_frequency": 0.0,
                }
            )
        return features

    def extract_sentence(self, sentence: str) -> list[dict[str, object]]:
        tokens = tokenize(sentence)
        context = self.context_from_tokens(tokens)
        return [
            self.extract_for_token(token, index, context)
            for index, token in enumerate(tokens)
            if is_word(token)
        ]

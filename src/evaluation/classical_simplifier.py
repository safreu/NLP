"""Sentence simplification using the trained classical classifier."""

from __future__ import annotations

from typing import cast

from preprocessing.classical_features import FeatureExtractor
from preprocessing.classical_replacements import ReplacementDictionary
from preprocessing.classical_text import detokenize, is_word, tokenize
from preprocessing.classical_training_data import SIMPLIFY
from training.classical_model import SimplificationModel


class ClassicalSimplifier:
    def __init__(
        self,
        model: SimplificationModel,
        feature_extractor: FeatureExtractor,
        replacement_dictionary: ReplacementDictionary,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.replacement_dictionary = replacement_dictionary

    def simplify(self, sentence: str) -> str:
        tokens = tokenize(sentence)
        context = self.feature_extractor.context_from_tokens(tokens)
        word_indices = [index for index, token in enumerate(tokens) if is_word(token)]
        if not word_indices:
            return sentence
        features = [
            self.feature_extractor.extract_for_token(tokens[index], index, context)
            for index in word_indices
        ]
        labels = cast(list[str], self.model.predict(features))
        output_tokens = list(tokens)
        for index, label in zip(word_indices, labels, strict=True):
            replacement = self.replacement_dictionary.best_replacement(tokens[index])
            if label == SIMPLIFY and replacement is not None:
                output_tokens[index] = self._match_case(tokens[index], replacement)
        return cast(str, detokenize(output_tokens))

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.title()
        return replacement

from dataclasses import dataclass
from pathlib import Path

from config import ClassicalMLConfig
from data.dataset_loader import Pair
from preprocessing.classical_features import FeatureExtractor
from preprocessing.classical_replacements import ReplacementDictionary
from preprocessing.classical_training_data import (
    build_raw_training_examples,
    build_token_frequencies,
    collect_replacements,
    extract_features_for_examples,
)
from storage.json_store import write_json
from training.classical_model import SimplificationModel


@dataclass(frozen=True)
class ClassicalTrainingArtifacts:
    model: SimplificationModel
    feature_extractor: FeatureExtractor
    replacement_dictionary: ReplacementDictionary
    validation_metrics: dict[str, float]


def train_classical_model(
    train_pairs: list[Pair],
    valid_pairs: list[Pair],
    path: Path,
    config: ClassicalMLConfig,
) -> ClassicalTrainingArtifacts:
    path.mkdir(parents=True, exist_ok=True)

    token_frequencies = build_token_frequencies(train_pairs, lowercase=config.lowercase)
    replacement_dictionary = collect_replacements(
        train_pairs,
        lowercase=config.lowercase,
        min_count=config.min_replacement_count,
    )
    feature_extractor = FeatureExtractor(
        token_frequencies=token_frequencies,
        replacement_dictionary=replacement_dictionary,
        lowercase=config.lowercase,
    )

    train_examples = extract_features_for_examples(
        build_raw_training_examples(train_pairs, lowercase=config.lowercase),
        feature_extractor,
    )
    valid_examples = extract_features_for_examples(
        build_raw_training_examples(valid_pairs, lowercase=config.lowercase),
        feature_extractor,
    )
    train_features = [example.features for example in train_examples]
    train_labels = [example.label for example in train_examples]
    valid_features = [example.features for example in valid_examples]
    valid_labels = [example.label for example in valid_examples]

    if not train_features:
        raise RuntimeError("No classical training features were created.")
    if len(set(train_labels)) < 2:
        raise RuntimeError("Classical training labels contain only one class.")

    model = SimplificationModel(
        model_type=config.model_type,
        random_state=config.random_state,
        classifier_parameters=config.classifier_parameters,
    )
    model.fit(train_features, train_labels)

    validation_metrics = (
        model.metrics(valid_features, valid_labels) if valid_features and valid_labels else {}
    )

    model.save(path / "model.pkl")
    replacement_dictionary.save(path / "replacement_dictionary.json")
    write_json(dict(token_frequencies), path / "token_frequencies.json")
    write_json(config_payload(config), path / "config.json")

    return ClassicalTrainingArtifacts(
        model=model,
        feature_extractor=feature_extractor,
        replacement_dictionary=replacement_dictionary,
        validation_metrics=validation_metrics,
    )


def config_payload(config: ClassicalMLConfig) -> dict[str, object]:
    return {
        "model_type": config.model_type,
        "random_state": config.random_state,
        "lowercase": config.lowercase,
        "min_replacement_count": config.min_replacement_count,
        "max_train_samples": config.max_train_samples,
        "max_eval_samples": config.max_eval_samples,
        "classifier_parameters": config.classifier_parameters or {},
        "compute_generation_metrics": config.compute_generation_metrics,
    }

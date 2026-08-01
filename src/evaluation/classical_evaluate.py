from collections.abc import Mapping
from pathlib import Path

from config import ClassicalMLConfig
from data.dataset_loader import Pair
from evaluation.classical_simplifier import ClassicalSimplifier
from evaluation.metrics_builder import compute_all_metrics
from preprocessing.classical_training_data import (
    build_raw_training_examples,
    extract_features_for_examples,
)
from storage.json_store import write_json
from storage.prediction_store import prediction_rows
from training.classical_trainer import ClassicalTrainingArtifacts


def evaluate_classical_model(
    test_pairs: list[Pair],
    artifacts: ClassicalTrainingArtifacts,
    predictions_path: Path,
    config: ClassicalMLConfig,
    extra_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    simplifier = ClassicalSimplifier(
        model=artifacts.model,
        feature_extractor=artifacts.feature_extractor,
        replacement_dictionary=artifacts.replacement_dictionary,
    )
    sources = [source for source, _ in test_pairs]
    references = [target for _, target in test_pairs]
    candidates = [simplifier.simplify(source) for source in sources]

    write_json(prediction_rows(sources, candidates, references), predictions_path)

    classifier_metrics: Mapping[str, object]
    if extra_metrics is not None:
        classifier_metrics = extra_metrics
    else:
        raw_examples = build_raw_training_examples(test_pairs, lowercase=config.lowercase)
        examples = extract_features_for_examples(raw_examples, artifacts.feature_extractor)
        classifier_metrics = artifacts.model.metrics(
            [example.features for example in examples],
            [example.label for example in examples],
        )

    metrics: dict[str, object] = {
        "classifier": dict(classifier_metrics),
    }
    if not config.compute_generation_metrics:
        metrics["prediction_count"] = len(candidates)
        return metrics

    metrics.update(compute_all_metrics(sources, candidates, references))
    return metrics

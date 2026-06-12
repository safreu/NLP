from pathlib import Path
from typing import cast

from config import ClassicalMLConfig
from data.dataset_loader import Pair
from evaluation.classical_simplifier import ClassicalSimplifier
from evaluation.metrics_builder import compute_all_metrics
from storage.json_store import write_json
from storage.prediction_store import prediction_rows
from training.classical_trainer import ClassicalTrainingArtifacts


def evaluate_classical_model(
    test_pairs: list[Pair],
    artifacts: ClassicalTrainingArtifacts,
    predictions_path: Path,
    config: ClassicalMLConfig,
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

    if not config.compute_generation_metrics:
        return {"prediction_count": len(candidates)}

    return cast(dict[str, object], compute_all_metrics(sources, candidates, references))

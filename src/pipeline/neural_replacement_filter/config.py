# mypy: ignore-errors
"""Configuration defaults for replacement-level candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = find_project_root()
NEURAL_FILTER_DIR = PROJECT_ROOT / "results" / "neural_replacement_filter"


@dataclass(frozen=True)
class CandidateBuildConfig:
    """Paths and knobs used by the debug candidate builder."""

    model_source: str = "logistic_regression"
    debug_size: int = 300
    output_dir: Path = NEURAL_FILTER_DIR / "outputs"
    prediction_data_path: Path = (
        PROJECT_ROOT
        / "present"
        / "Predicted_sentences"
        / "logistic_regression_wikismall_final_predictions.csv"
    )
    replacement_dictionary_path: Path = (
        PROJECT_ROOT
        / "classical+logistical reg"
        / "outputs"
        / "models"
        / "classical_ml_wikismall"
        / "replacement_dictionary.json"
    )


DEFAULT_CONFIG = CandidateBuildConfig()


OUTPUT_FILENAMES = {
    "all_candidates": "candidate_replacements_debug.csv",
    "blocked": "blocked_replacements_debug.csv",
    "uncertain": "uncertain_replacements_debug.csv",
    "training": "neural_training_candidates_debug.csv",
    "feature_preview": "feature_preview_debug.csv",
    "label_distribution": "label_distribution_debug.json",
    "examples": "debug_examples.txt",
}

OUTPUT_DIR = NEURAL_FILTER_DIR / "outputs"
LOG_DIR = NEURAL_FILTER_DIR / "logs"
MODEL_DIR = NEURAL_FILTER_DIR / "models"

TRAINING_CANDIDATES_PATH = OUTPUT_DIR / "neural_training_candidates_debug.csv"
ALL_CANDIDATES_PATH = OUTPUT_DIR / "candidate_replacements_debug.csv"
NEURAL_MODEL_PATH = MODEL_DIR / "neural_filter_debug.pt"
FEATURE_PREPROCESSOR_PATH = MODEL_DIR / "feature_preprocessor_debug.pkl"
TRAINING_METRICS_PATH = OUTPUT_DIR / "neural_training_metrics_debug.json"
TRAINING_LOG_PATH = LOG_DIR / "neural_training_debug.log"

CATEGORICAL_FEATURES = ["model_source", "pos_tag", "ner_tag"]
NUMERIC_FEATURES = [
    "model_confidence",
    "is_proper_noun",
    "is_number",
    "is_connector",
    "original_length",
    "replacement_length",
    "length_difference",
]
FEATURE_COLUMNS = [
    "model_confidence",
    "model_source",
    "pos_tag",
    "ner_tag",
    "is_proper_noun",
    "is_number",
    "is_connector",
    "original_length",
    "replacement_length",
    "length_difference",
]

THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
RANDOM_SEED = 42

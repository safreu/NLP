from statistics import mean

from evaluation.analyzers.analyzer_utils import (
    normalize_text,
    token_f1,
)
from evaluation.analyzers.base import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


class CopyAnalyzer(PredictionAnalyzer):
    """
    Analyzes copy behavior between source sentences and generated candidates.

    Each JSON row represents a single prediction and contains:
    - the original source sentence,
    - the generated candidate,
    - the human reference simplification,
    - the similarity score between source and candidate,
    - and the assigned copy category.

    Predictions are classified as:
    - exact copies: candidate equals source after normalization,
    - near copies: candidate similarity exceeds the configured threshold,
    - different predictions: candidate sufficiently differs from the source.

    The generated summary additionally reports dataset-level copy ratios
    and overall copy statistics.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        rows: list[dict[str, object]] = []
        exact_copies: list[dict[str, object]] = []
        near_copies: list[dict[str, object]] = []
        different_predictions: list[dict[str, object]] = []

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            source_overlap = token_f1(candidate, source)
            reference_overlap = token_f1(candidate, reference)

            exact_copy = normalize_text(source) == normalize_text(candidate)

            result = {
                "index": index,
                "source": source,
                "candidate": candidate,
                "reference": reference,
                "source_token_overlap": source_overlap,
                "reference_token_overlap": reference_overlap,
                "exact_copy": exact_copy,
                "changed_from_source": not exact_copy,
            }

            rows.append(result)

            if exact_copy:
                exact_copies.append(result)
            elif source_overlap >= self.threshold:
                near_copies.append(result)
            else:
                different_predictions.append(result)

        total = len(predictions)

        summary = {
            "num_predictions": total,
            "exact_copy_count": len(exact_copies),
            "near_copy_count": len(near_copies),
            "different_count": len(different_predictions),
            "exact_copy_ratio": len(exact_copies) / total if total else 0.0,
            "near_copy_ratio": len(near_copies) / total if total else 0.0,
            "different_ratio": len(different_predictions) / total if total else 0.0,
            "percent_sentences_changed": (
                sum(bool(row["changed_from_source"]) for row in rows) / total * 100
                if total
                else 0.0
            ),
            "avg_source_token_overlap": (
                mean(float(row["source_token_overlap"]) for row in rows) if rows else 0.0
            ),
            "avg_reference_token_overlap": (
                mean(float(row["reference_token_overlap"]) for row in rows) if rows else 0.0
            ),
            "near_copy_threshold": self.threshold,
        }

        report = {
            "summary": summary,
            "data": rows,
            "exact_copies": exact_copies,
            "near_copies": near_copies,
            "different_predictions": different_predictions,
        }

        write_json(report, run_paths.copy_analysis_path)

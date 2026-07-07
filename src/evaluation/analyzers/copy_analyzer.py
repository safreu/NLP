from evaluation.analyzers.base import PredictionAnalyzer
from preprocessing import filter
from preprocessing.cleaner import normalize_text
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

    def __init__(self, threshold):
        self.threshold = threshold

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        exact_copies = []
        near_copies = []
        different_predictions = []

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            similarity = filter.text_similarity(source, candidate)

            result = {
                "index": index,
                "similarity": similarity,
                "source": source,
                "candidate": candidate,
                "reference": reference,
            }

            if normalize_text(source) == normalize_text(candidate):
                exact_copies.append(result)
            elif similarity >= self.threshold:
                near_copies.append(result)
            else:
                different_predictions.append(result)

        total = len(predictions)

        report = {
            "num_predictions": total,
            "exact_copy_count": len(exact_copies),
            "near_copy_count": len(near_copies),
            "different_count": len(different_predictions),
            "exact_copy_ratio": len(exact_copies) / total if total else 0.0,
            "near_copy_ratio": len(near_copies) / total if total else 0.0,
            "different_ratio": len(different_predictions) / total if total else 0.0,
            "near_copy_threshold": self.threshold,
            "exact_copies": exact_copies,
            "near_copies": near_copies,
            "different_predictions": different_predictions,
        }

        write_json(report, run_paths.copy_analysis_path)

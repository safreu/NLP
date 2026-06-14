from statistics import mean

from evaluation.analyzers.base import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


def word_count(text: str) -> int:
    return len(text.split())


def safe_ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


class LengthAnalyzer(PredictionAnalyzer):
    """
    Analyzes length and compression behavior of generated simplifications.

    Each JSON row represents a single prediction and contains:
    - source sentence word count,
    - candidate sentence word count,
    - reference sentence word count,
    - candidate-to-source length ratio,
    - and candidate-to-reference length ratio.

    The generated summary reports average sentence lengths and average
    compression ratios across the entire dataset.
    """

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        rows = []

        for index, row in enumerate(predictions):
            source_len = word_count(row["source"])
            candidate_len = word_count(row["candidate"])
            reference_len = word_count(row["reference"])

            rows.append(
                {
                    "index": index,
                    "source_word_count": source_len,
                    "candidate_word_count": candidate_len,
                    "reference_word_count": reference_len,
                    "candidate_source_ratio": safe_ratio(candidate_len, source_len),
                    "candidate_reference_ratio": safe_ratio(candidate_len, reference_len),
                }
            )

        summary = {
            "num_predictions": len(rows),
            "avg_source_word_count": mean([r["source_word_count"] for r in rows]) if rows else 0,
            "avg_candidate_word_count": mean([r["candidate_word_count"] for r in rows])
            if rows
            else 0,
            "avg_reference_word_count": mean([r["reference_word_count"] for r in rows])
            if rows
            else 0,
            "avg_candidate_source_ratio": mean([r["candidate_source_ratio"] for r in rows])
            if rows
            else 0,
            "avg_candidate_reference_ratio": mean([r["candidate_reference_ratio"] for r in rows])
            if rows
            else 0,
        }

        write_json({"summary": summary, "data": rows}, run_paths.length_analysis_path)

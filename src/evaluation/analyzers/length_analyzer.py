from statistics import mean

from evaluation.analyzers.analyzer_utils import (
    safe_ratio,
    tokens,
    words,
)
from evaluation.analyzers.base import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


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
        rows: list[dict[str, object]] = []

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            source_tokens = tokens(source)
            candidate_tokens = tokens(candidate)
            reference_tokens = tokens(reference)

            source_words = words(source)
            candidate_words = words(candidate)
            reference_words = words(reference)

            rows.append(
                {
                    "index": index,
                    "source": source,
                    "candidate": candidate,
                    "reference": reference,
                    "source_token_count": len(source_tokens),
                    "candidate_token_count": len(candidate_tokens),
                    "reference_token_count": len(reference_tokens),
                    "source_sentence_length": len(source),
                    "candidate_sentence_length": len(candidate),
                    "reference_sentence_length": len(reference),
                    "source_avg_word_length": (
                        sum(len(word) for word in source_words) / len(source_words)
                        if source_words
                        else 0.0
                    ),
                    "candidate_avg_word_length": (
                        sum(len(word) for word in candidate_words) / len(candidate_words)
                        if candidate_words
                        else 0.0
                    ),
                    "reference_avg_word_length": (
                        sum(len(word) for word in reference_words) / len(reference_words)
                        if reference_words
                        else 0.0
                    ),
                    "candidate_source_ratio": safe_ratio(len(candidate_tokens), len(source_tokens)),
                    "candidate_reference_ratio": safe_ratio(
                        len(candidate_tokens), len(reference_tokens)
                    ),
                }
            )

        summary = {
            "num_predictions": len(rows),
            "avg_source_token_count": (
                mean(float(row["source_token_count"]) for row in rows) if rows else 0.0
            ),
            "avg_candidate_token_count": (
                mean(float(row["candidate_token_count"]) for row in rows) if rows else 0.0
            ),
            "avg_reference_token_count": (
                mean(float(row["reference_token_count"]) for row in rows) if rows else 0.0
            ),
            "avg_source_sentence_length": (
                mean(float(row["source_sentence_length"]) for row in rows) if rows else 0.0
            ),
            "avg_candidate_sentence_length": (
                mean(float(row["candidate_sentence_length"]) for row in rows) if rows else 0.0
            ),
            "avg_reference_sentence_length": (
                mean(float(row["reference_sentence_length"]) for row in rows) if rows else 0.0
            ),
            "avg_source_word_length": (
                mean(float(row["source_avg_word_length"]) for row in rows) if rows else 0.0
            ),
            "avg_candidate_word_length": (
                mean(float(row["candidate_avg_word_length"]) for row in rows) if rows else 0.0
            ),
            "avg_reference_word_length": (
                mean(float(row["reference_avg_word_length"]) for row in rows) if rows else 0.0
            ),
            "avg_candidate_source_ratio": (
                mean(float(row["candidate_source_ratio"]) for row in rows) if rows else 0.0
            ),
            "avg_candidate_reference_ratio": (
                mean(float(row["candidate_reference_ratio"]) for row in rows) if rows else 0.0
            ),
        }

        write_json({"summary": summary, "data": rows}, run_paths.length_analysis_path)

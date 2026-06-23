from statistics import mean

from evaluation.analyzers.base import PredictionAnalyzer
from metrics.flesch_kincaid import compute_flesch_kincaid_score
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


class ReadabilityAnalyzer(PredictionAnalyzer):
    """
    Analyzes Flesch-Kincaid readability grade levels for source,
    generated candidate, and human reference texts.

    Each JSON row represents a single prediction and contains:
    - the source sentence readability score,
    - the generated candidate readability score,
    - the human reference readability score,
    - the candidate-source readability difference,
    - the reference-source readability difference,
    - and the candidate-reference readability difference.

    Lower Flesch-Kincaid scores usually indicate easier text. The summary
    reports average readability before and after simplification and compares
    the model output against the human reference.
    """

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        rows = []

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            source_score = compute_flesch_kincaid_score(source)
            candidate_score = compute_flesch_kincaid_score(candidate)
            reference_score = compute_flesch_kincaid_score(reference)

            rows.append(
                {
                    "index": index,
                    "source": source,
                    "candidate": candidate,
                    "reference": reference,
                    "source_flesch_kincaid": source_score,
                    "candidate_flesch_kincaid": candidate_score,
                    "reference_flesch_kincaid": reference_score,
                    "candidate_source_delta": candidate_score - source_score,
                    "reference_source_delta": reference_score - source_score,
                    "candidate_reference_delta": candidate_score - reference_score,
                }
            )

        summary = {
            "num_predictions": len(rows),
            "avg_source_flesch_kincaid": mean([row["source_flesch_kincaid"] for row in rows])
            if rows
            else 0.0,
            "avg_candidate_flesch_kincaid": mean([row["candidate_flesch_kincaid"] for row in rows])
            if rows
            else 0.0,
            "avg_reference_flesch_kincaid": mean([row["reference_flesch_kincaid"] for row in rows])
            if rows
            else 0.0,
            "avg_candidate_source_delta": mean([row["candidate_source_delta"] for row in rows])
            if rows
            else 0.0,
            "avg_reference_source_delta": mean([row["reference_source_delta"] for row in rows])
            if rows
            else 0.0,
            "avg_candidate_reference_delta": mean(
                [row["candidate_reference_delta"] for row in rows]
            )
            if rows
            else 0.0,
        }

        write_json(
            {"summary": summary, "data": rows},
            run_paths.readability_analysis_path,
        )

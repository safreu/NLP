from collections import Counter

from evaluation.analyzers.prediction_analyzer import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


class DiversityAnalyzer(PredictionAnalyzer):
    def run(self, predictions: list[PredictionRow], run_paths: RunPaths):
        candidates = [row["candidate"].strip() for row in predictions]
        candidate_counter = Counter(candidates)

        repeated = [
            {
                "candidate": candidate,
                "count": count,
            }
            for candidate, count in candidate_counter.most_common()
            if count > 1
        ]

        empty_outputs = [index for index, candidate in enumerate(candidates) if not candidate]

        summary = {
            "num_predictions": len(candidates),
            "unique_candidate_count": len(candidate_counter),
            "unique_candidate_ratio": len(candidate_counter) / len(candidates)
            if candidates
            else 0.0,
            "repeated_candidate_count": len(repeated),
            "empty_output_count": len(empty_outputs),
            "most_common_candidates": candidate_counter.most_common(25),
        }

        write_json(
            {
                "summary": summary,
                "repeated_candidates": repeated,
                "empty_output_indices": empty_outputs,
            },
            run_paths.diversity_analysis_path,
        )

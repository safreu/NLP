from collections import Counter

from evaluation.analyzers.analyzer_utils import normalize_text
from evaluation.analyzers.base import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


class DiversityAnalyzer(PredictionAnalyzer):
    """
    Analyzes output diversity across generated simplifications.

    Each JSON row represents repeated or problematic candidate generations
    and contains:
    - the generated candidate text,
    - and the number of occurrences within the dataset.

    The analyzer additionally tracks:
    - repeated outputs,
    - empty generations,
    - the number of unique candidates,
    - and overall diversity ratios.

    """

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        candidates = [row["candidate"].strip() for row in predictions]
        
        normalized_candidates = [
            normalize_text(candidate)
            for candidate in candidates    
        ]
        
        raw_counter = Counter(candidates)
        normalized_counter = Counter(normalized_candidates)
        

        repeated = [
            {
                "candidate": candidate,
                "count": count,
            }
            for candidate, count in raw_counter.most_common()
            if count > 1
        ]
        
        normalized_repeated = [
            {
                "normalized_candidate": candidate,
                "count": count,
            }
            for candidate, count in normalized_counter.most_common()
            if count > 1
        ]

        empty_outputs = [index for index, candidate in enumerate(candidates) if not candidate]
        
        total = len(candidates)

        summary = {
            "num_predictions": total,
            "unique_candidate_count": len(raw_counter),
            "unique_candidate_ratio": len(raw_counter) / total if total else 0.0,
            "normalized_unique_candidate_count": len(normalized_counter),
            "normalized_unique_candidate_ratio": len(normalized_counter) / total if total else 0.0,
            "repeated_candidate_count": len(repeated),
            "normalized_repeated_candidate_count": len(normalized_repeated),
            "empty_output_count": len(empty_outputs),
            "most_common_candidates": raw_counter.most_common(25),
            "most_common_normalized_candidates": normalized_counter.most_common(25),
        }

        write_json(
            {
                "summary": summary,
                "repeated_candidates": repeated,
                "normalized_repeated_candidates": normalized_repeated,
                "empty_output_indices": empty_outputs,
            },
            run_paths.diversity_analysis_path,
        )

from collections import Counter

from evaluation.analyzers.analyzer_utils import (
    normalize_text,
    safe_ratio,
    tokens,
)
from evaluation.analyzers.base import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


class ErrorCaseAnalyzer(PredictionAnalyzer):
    """
    Analyzes problematic or unusual model generations.

    Each JSON row represents a detected error case and contains:
    - the source sentence,
    - the generated candidate,
    - the human reference simplification,
    - assigned error labels,
    - and sentence length statistics.

    Tracked error categories include:
    - empty candidates,
    - exact source copies,
    - candidates longer than the source,
    - and excessively short candidates.

    The generated summary reports overall error frequencies and dataset-level
    error ratios.
    """

    def __init__(self, long_ratio: float = 1.2, short_ratio: float = 0.3) -> None:
        self.long_ratio = long_ratio
        self.short_ratio = short_ratio

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        cases: list[dict[str, object]] = []
        label_counts: Counter[str] = Counter()

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            source_count = len(tokens(source))
            candidate_count = len(tokens(candidate))

            length_ratio = safe_ratio(candidate_count, source_count)

            labels: list[str] = []

            if not candidate.strip():
                labels.append("empty_candidate")

            if normalize_text(source) == normalize_text(candidate):
                labels.append("exact_copy")

            if source_count and length_ratio > self.long_ratio:
                labels.append("candidate_longer_than_source")

            if source_count and length_ratio < self.short_ratio:
                labels.append("candidate_very_short")

            if labels:
                label_counts.update(labels)

                cases.append(
                    {
                        "index": index,
                        "labels": labels,
                        "source": source,
                        "candidate": candidate,
                        "reference": reference,
                        "source_token_count": source_count,
                        "candidate_token_count": candidate_count,
                        "candidate_source_ratio": length_ratio,
                    }
                )

        total = len(predictions)

        summary = {
            "num_predictions": total,
            "num_error_cases": len(cases),
            "error_case_ratio": len(cases) / total if total else 0.0,
            "label_counts": dict(label_counts),
            "label_ratios": {
                label: (count / total if total else 0.0) for label, count in label_counts.items()
            },
            "long_ratio_threshold": self.long_ratio,
            "short_ratio_threshold": self.short_ratio,
        }

        write_json(
            {
                "summary": summary,
                "data": cases,
            },
            run_paths.error_case_analysis_path,
        )

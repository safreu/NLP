from evaluation.analyzers.prediction_analyzer import PredictionAnalyzer
from preprocessing.cleaner import normalize_text
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


def word_count(text: str) -> int:
    return len(text.split())


class ErrorCaseAnalyzer(PredictionAnalyzer):
    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        cases = []

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            source_len = word_count(source)
            candidate_len = word_count(candidate)

            labels = []

            if not candidate.strip():
                labels.append("empty_candidate")

            if normalize_text(source) == normalize_text(candidate):
                labels.append("exact_copy")

            if source_len and candidate_len / source_len > 1.2:
                labels.append("candidate_longer_than_source")

            if source_len and candidate_len / source_len < 0.3:
                labels.append("candidate_very_short")

            if labels:
                cases.append(
                    {
                        "index": index,
                        "labels": labels,
                        "source": source,
                        "candidate": candidate,
                        "reference": reference,
                        "source_word_count": source_len,
                        "candidate_word_count": candidate_len,
                    }
                )

        summary = {
            "num_predictions": len(predictions),
            "num_error_cases": len(cases),
            "error_case_ratio": len(cases) / len(predictions) if predictions else 0.0,
        }

        write_json(
            {
                "summary": summary,
                "data": cases,
            },
            run_paths.error_case_analysis_path,
        )

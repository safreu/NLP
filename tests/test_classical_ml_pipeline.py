from pathlib import Path
from types import SimpleNamespace

from configuration.classic_ml_config import ClassicalMLConfig
from pipeline import classical_ml_pipeline
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow, write_predictions


class StubLoader:
    def load_pairs(self):
        return (
            [("Training source.", "Training target.")],
            [("Validation source.", "Validation target.")],
            [("Test source.", "Test target.")],
        )


class StubAnalyzer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[PredictionRow], RunPaths]] = []

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        self.calls.append((predictions, run_paths))


def test_classical_pipeline_runs_analyzers_on_saved_predictions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifacts = SimpleNamespace(validation_metrics={"accuracy": 0.8})
    analyzer = StubAnalyzer()

    monkeypatch.setattr(
        classical_ml_pipeline,
        "train_classical_model",
        lambda train_pairs, valid_pairs, path, config: artifacts,
    )

    def stub_evaluate(test_pairs, artifacts, predictions_path, config):
        write_predictions(
            sources=["Test source."],
            candidates=["Simplified test."],
            references=["Test target."],
            path=predictions_path,
        )
        return {"sari": 42.0}

    monkeypatch.setattr(
        classical_ml_pipeline,
        "evaluate_classical_model",
        stub_evaluate,
    )

    pipeline = classical_ml_pipeline.ClassicalMLPipeline(
        name="classical",
        dataset_loader=StubLoader(),
        config=ClassicalMLConfig(),
        analyzers=[analyzer],
    )
    pipeline.run(tmp_path)

    pipeline_dir = tmp_path / "classical"
    assert analyzer.calls == [
        (
            [
                {
                    "source": "Test source.",
                    "candidate": "Simplified test.",
                    "reference": "Test target.",
                }
            ],
            RunPaths(pipeline_dir),
        )
    ]

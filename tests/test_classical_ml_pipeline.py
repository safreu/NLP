from pathlib import Path
from types import SimpleNamespace

from config import ClassicalMLConfig
from pipeline import classical_ml_pipeline
from storage.json_store import read_json


class StubLoader:
    name = "stub"

    def load_pairs(self):
        return (
            [("The physician administered medication.", "The doctor gave medicine.")],
            [("The automobile stopped.", "The car stopped.")],
            [("The residence was large.", "The house was big.")],
        )


class StubSimplifier:
    def __init__(self, model, feature_extractor, replacement_dictionary) -> None:
        pass

    def simplify(self, sentence: str) -> str:
        return f"simple: {sentence}"


def test_classical_pipeline_scores_validation_and_test_generation_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifacts = SimpleNamespace(
        model=object(),
        feature_extractor=object(),
        replacement_dictionary=object(),
        validation_metrics={"accuracy": 0.8, "macro_f1": 0.75},
    )
    metric_calls: list[tuple[list[str], list[str], list[str]]] = []

    def stub_train_classical_model(train_pairs, valid_pairs, path, config):
        assert train_pairs == [
            ("The physician administered medication.", "The doctor gave medicine.")
        ]
        assert valid_pairs == [("The automobile stopped.", "The car stopped.")]
        return artifacts

    def stub_compute_all_metrics(sources, candidates, references):
        metric_calls.append((sources, candidates, references))
        return {
            "bert": {"f1_mean": 0.9},
            "bleu": 0.3,
            "f1": {"f1_mean": 0.7},
            "flesch": {"mean": 4.2},
            "sari": 41.5,
            "rouge-l": 0.62,
        }

    monkeypatch.setattr(classical_ml_pipeline, "train_classical_model", stub_train_classical_model)
    monkeypatch.setattr("evaluation.classical_evaluate.ClassicalSimplifier", StubSimplifier)
    monkeypatch.setattr(
        "evaluation.classical_evaluate.compute_all_metrics",
        stub_compute_all_metrics,
    )

    pipeline = classical_ml_pipeline.ClassicalMLPipeline(
        name="classical",
        dataset_loader=StubLoader(),
        config=ClassicalMLConfig(),
    )

    pipeline.run(tmp_path)

    pipeline_dir = tmp_path / "classical"
    scores = read_json(pipeline_dir / "scores.json")
    validation_predictions = read_json(pipeline_dir / "validation_predictions.json")
    test_predictions = read_json(pipeline_dir / "predictions.json")

    assert scores["validation"]["accuracy"] == 0.8
    assert scores["validation"]["macro_f1"] == 0.75
    for split in ("validation", "test"):
        assert {"bert", "bleu", "f1", "flesch", "sari", "rouge-l"} <= set(scores[split])

    assert validation_predictions == [
        {
            "source": "The automobile stopped.",
            "candidate": "simple: The automobile stopped.",
            "reference": "The car stopped.",
        }
    ]
    assert test_predictions == [
        {
            "source": "The residence was large.",
            "candidate": "simple: The residence was large.",
            "reference": "The house was big.",
        }
    ]
    assert metric_calls == [
        (
            ["The automobile stopped."],
            ["simple: The automobile stopped."],
            ["The car stopped."],
        ),
        (
            ["The residence was large."],
            ["simple: The residence was large."],
            ["The house was big."],
        ),
    ]


def test_classical_pipeline_lightweight_validation_keeps_classifier_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifacts = SimpleNamespace(
        model=object(),
        feature_extractor=object(),
        replacement_dictionary=object(),
        validation_metrics={"accuracy": 0.8, "macro_f1": 0.75},
    )

    monkeypatch.setattr(
        classical_ml_pipeline,
        "train_classical_model",
        lambda train_pairs, valid_pairs, path, config: artifacts,
    )
    monkeypatch.setattr("evaluation.classical_evaluate.ClassicalSimplifier", StubSimplifier)

    pipeline = classical_ml_pipeline.ClassicalMLPipeline(
        name="classical",
        dataset_loader=StubLoader(),
        config=ClassicalMLConfig(compute_generation_metrics=False),
    )

    pipeline.run(tmp_path)

    scores = read_json(tmp_path / "classical" / "scores.json")

    assert scores["validation"] == {
        "accuracy": 0.8,
        "macro_f1": 0.75,
        "prediction_count": 1,
    }
    assert scores["test"] == {"prediction_count": 1}

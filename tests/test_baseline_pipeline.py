from pathlib import Path

from evaluation.result_aggregation import aggregate_results
from pipeline import baseline_pipeline
from prompts import simplify_prompt
from storage.json_store import read_json


def test_copy_baseline_removes_prompt_and_normalizes_whitespace() -> None:
    prediction = baseline_pipeline.copy_baseline(
        simplify_prompt("The physician   administered medication.\n")
    )

    assert prediction == "The physician administered medication."


def test_punctuation_split_baseline_applies_simple_rules() -> None:
    prediction = baseline_pipeline.punctuation_split_baseline(
        "The physician administered medication; the patient recovered, which pleased everyone"
    )

    assert prediction == (
        "The physician administered medication. the patient recovered. which pleased everyone."
    )


def test_selected_baselines_defaults_to_all_baselines() -> None:
    baselines = baseline_pipeline.selected_baselines(None)

    assert [baseline.name for baseline in baselines] == ["copy", "punctuation_split"]


def test_load_test_pairs_caps_wikilarge_loader(
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, int | None] = {}

    class StubWikiLargeLoader:
        def __init__(self, max_train_samples, max_eval_samples):
            captured_kwargs["max_train_samples"] = max_train_samples
            captured_kwargs["max_eval_samples"] = max_eval_samples

        def load_pairs(self):
            return [], [], [("source 1", "reference 1"), ("source 2", "reference 2")]

    monkeypatch.setattr(baseline_pipeline, "WikiLargeLoader", StubWikiLargeLoader)

    test_pairs = baseline_pipeline.load_test_pairs("wikilarge", 1)

    assert captured_kwargs == {"max_train_samples": 0, "max_eval_samples": 1}
    assert test_pairs == [("source 1", "reference 1")]


def test_run_baselines_writes_scores_predictions_and_aggregate_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def stub_load_test_pairs(dataset_name: str, max_examples: int):
        assert dataset_name == "wikilarge"
        assert max_examples == 2
        return [
            ("The physician administered medication.", "The doctor gave medicine."),
            ("The patient recovered; everyone was pleased.", "The patient got better."),
        ]

    def stub_compute_all_metrics(sources, predictions, references):
        assert sources == [
            "The physician administered medication.",
            "The patient recovered; everyone was pleased.",
        ]
        assert references == ["The doctor gave medicine.", "The patient got better."]
        return {
            "bert": {"precision_mean": 0.9, "recall_mean": 0.8, "f1_mean": 0.85},
            "bleu": 0.3,
            "f1": {"f1_mean": 0.7},
            "flesch": {"mean": 5.1},
            "sari": 39.5,
            "rouge-l": 0.61,
            "prediction_count": len(predictions),
        }

    monkeypatch.setattr(baseline_pipeline, "load_test_pairs", stub_load_test_pairs)
    monkeypatch.setattr(baseline_pipeline, "compute_all_metrics", stub_compute_all_metrics)

    output_path = tmp_path / "baseline_run"
    args = baseline_pipeline.parse_args(
        [
            "--dataset",
            "wikilarge",
            "--baseline",
            "copy",
            "--max-examples",
            "2",
            "--output-path",
            str(output_path),
        ]
    )

    run_dir = baseline_pipeline.run_baselines(args)

    assert run_dir == output_path
    assert read_json(output_path / "config.json") == {
        "datasets": ["wikilarge"],
        "baselines": [
            {
                "name": "copy",
                "description": "Copies the source sentence after removing any training prompt.",
            }
        ],
        "max_examples": 2,
    }

    pipeline_dir = output_path / "wikilarge_copy"
    predictions = read_json(pipeline_dir / "predictions.json")
    scores = read_json(pipeline_dir / "scores.json")
    rows = aggregate_results([output_path])

    assert predictions[0] == {
        "source": "The physician administered medication.",
        "candidate": "The physician administered medication.",
        "reference": "The doctor gave medicine.",
    }
    assert scores["sari"] == 39.5
    assert rows[0]["run"] == "baseline_run"
    assert rows[0]["pipeline"] == "wikilarge_copy"
    assert rows[0]["sari"] == 39.5

from pathlib import Path

from evaluation import result_aggregation
from storage.json_store import write_json


def write_score_file(path: Path, data: object) -> None:
    write_json(data, path)


def write_training_run_config(run_dir: Path, pipeline_name: str, model_name: str) -> None:
    write_json(
        {
            "pipelines": [
                {
                    "name": pipeline_name,
                    "training_config": {"model_name": model_name},
                }
            ]
        },
        run_dir / "config.json",
    )


def write_baseline_run_config(run_dir: Path, baselines: object) -> None:
    write_json(
        {
            "datasets": ["onestop", "wikilarge"],
            "baselines": baselines,
            "max_examples": None,
        },
        run_dir / "config.json",
    )


def metric_payload() -> dict[str, object]:
    return {
        "bert": {
            "precision_mean": 0.91,
            "recall_mean": 0.82,
            "f1_mean": 0.86,
        },
        "bleu": 0.32,
        "f1": {"f1_mean": 0.75},
        "flesch": {"mean": 4.2},
        "sari": 41.5,
        "rouge-l": 0.62,
    }


def test_aggregate_final_model_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    write_training_run_config(run_dir, "onestop", "google/flan-t5-base")
    scores_path = run_dir / "onestop" / "scores.json"
    write_score_file(scores_path, metric_payload())

    rows = result_aggregation.aggregate_results([run_dir])

    assert rows == [
        {
            "run": "run_001",
            "pipeline": "onestop",
            "model": "google/flan-t5-base",
            "checkpoint": "",
            "sari": 41.5,
            "fkgl": 4.2,
            "bertscore_precision": 0.91,
            "bertscore_recall": 0.82,
            "bertscore_f1": 0.86,
            "bleu": 0.32,
            "rouge_l": 0.62,
            "token_f1": 0.75,
            "scores_path": str(scores_path),
        }
    ]


def test_aggregate_direct_scores_file_infers_pipeline_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    write_training_run_config(run_dir, "onestop", "google/flan-t5-base")
    scores_path = run_dir / "onestop" / "scores.json"
    write_score_file(scores_path, metric_payload())

    rows = result_aggregation.aggregate_results([scores_path])

    assert rows[0]["run"] == "run_001"
    assert rows[0]["pipeline"] == "onestop"
    assert rows[0]["model"] == "google/flan-t5-base"


def test_aggregate_baseline_direct_scores_file_resolves_copy_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "baselines_full"
    write_baseline_run_config(
        run_dir,
        [
            {"name": "copy", "description": "Copies the source sentence."},
            {"name": "punctuation_split", "description": "Splits on punctuation."},
        ],
    )
    scores_path = run_dir / "wikilarge_copy" / "scores.json"
    write_score_file(scores_path, metric_payload())

    rows = result_aggregation.aggregate_results([scores_path])

    assert rows[0]["run"] == "wikilarge_copy"
    assert rows[0]["pipeline"] == ""
    assert rows[0]["model"] == "copy"


def test_aggregate_baseline_run_directory_resolves_punctuation_split_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "baselines_full"
    write_baseline_run_config(
        run_dir,
        [
            {"name": "split", "description": "Generic split baseline."},
            {"name": "punctuation_split", "description": "Splits on punctuation."},
        ],
    )
    pipeline_dir = run_dir / "wikilarge_punctuation_split"
    write_score_file(pipeline_dir / "scores.json", metric_payload())

    rows = result_aggregation.aggregate_results([pipeline_dir])

    assert rows[0]["run"] == "wikilarge_punctuation_split"
    assert rows[0]["pipeline"] == ""
    assert rows[0]["model"] == "punctuation_split"


def test_aggregate_parent_directory_resolves_multiple_baseline_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "baselines_full"
    write_baseline_run_config(
        run_dir,
        [
            {"name": "copy", "description": "Copies the source sentence."},
            {"name": "punctuation_split", "description": "Splits on punctuation."},
        ],
    )
    write_score_file(run_dir / "wikilarge_copy" / "scores.json", metric_payload())
    write_score_file(run_dir / "wikilarge_punctuation_split" / "scores.json", metric_payload())

    rows = result_aggregation.aggregate_results([run_dir])
    models_by_pipeline = {str(row["pipeline"]): row["model"] for row in rows}

    assert models_by_pipeline == {
        "wikilarge_copy": "copy",
        "wikilarge_punctuation_split": "punctuation_split",
    }


def test_aggregate_checkpoint_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_002"
    write_training_run_config(run_dir, "wikilarge", "facebook/bart-base")
    scores_path = run_dir / "wikilarge" / "scores.json"
    write_score_file(
        scores_path,
        {
            "checkpoint-100": metric_payload(),
            "checkpoint-200": {"sari": 44.0, "bleu": 0.4},
        },
    )

    rows = result_aggregation.aggregate_results([run_dir])

    assert [row["checkpoint"] for row in rows] == ["checkpoint-100", "checkpoint-200"]
    assert rows[0]["pipeline"] == "wikilarge"
    assert rows[0]["model"] == "facebook/bart-base"
    assert rows[0]["sari"] == 41.5
    assert rows[1]["sari"] == 44.0
    assert rows[1]["model"] == "facebook/bart-base"
    assert rows[1]["fkgl"] is None
    assert rows[1]["bleu"] == 0.4


def test_invalid_baseline_configuration_keeps_empty_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "baselines_full"
    write_baseline_run_config(run_dir, ["copy", {"description": "Missing baseline name."}])
    scores_path = run_dir / "wikilarge_copy" / "scores.json"
    write_score_file(scores_path, metric_payload())

    rows = result_aggregation.aggregate_results([scores_path])

    assert rows[0]["run"] == "wikilarge_copy"
    assert rows[0]["model"] == ""


def test_aggregate_flat_legacy_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_003"
    write_training_run_config(run_dir, "onestop", "google/flan-t5-base")
    scores_path = run_dir / "scores.json"
    write_score_file(
        scores_path,
        {
            "bertscore_precision_mean": 0.9,
            "bertscore_recall_mean": 0.8,
            "bertscore_f1_mean": 0.85,
            "bleu": 0.3,
            "token_f1_mean": 0.7,
            "flesch_kincaid_mean": 5.1,
            "sari": 39.5,
            "rouge_l": 0.61,
            "asset_multi_reference_sari": 42.0,
        },
    )

    rows = result_aggregation.aggregate_results([run_dir])

    assert rows[0]["run"] == "run_003"
    assert rows[0]["pipeline"] == ""
    assert rows[0]["model"] == "google/flan-t5-base"
    assert rows[0]["sari"] == 39.5
    assert rows[0]["fkgl"] == 5.1
    assert rows[0]["bertscore_f1"] == 0.85
    assert rows[0]["rouge_l"] == 0.61
    assert rows[0]["token_f1"] == 0.7


def test_markdown_uses_missing_value_placeholder() -> None:
    markdown = result_aggregation.format_markdown(
        [
            {
                "run": "run_001",
                "pipeline": "onestop",
                "model": "google/flan-t5-base",
                "checkpoint": "",
                "sari": 41.5,
                "fkgl": None,
                "bertscore_precision": None,
                "bertscore_recall": None,
                "bertscore_f1": None,
                "bleu": 0.32,
                "rouge_l": None,
                "token_f1": None,
                "scores_path": "runs/run_001/onestop/scores.json",
            }
        ]
    )

    assert "| run | pipeline | model | checkpoint | sari | fkgl |" in markdown
    assert "| run_001 | onestop | google/flan-t5-base |  | 41.5000 | NA |" in markdown


def test_format_csv_and_json_outputs() -> None:
    rows: list[result_aggregation.ResultRow] = [
        {
            "run": "run_001",
            "pipeline": "onestop",
            "model": "google/flan-t5-base",
            "checkpoint": "",
            "sari": 41.5,
            "fkgl": None,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "bleu": 0.32,
            "rouge_l": None,
            "token_f1": None,
            "scores_path": "runs/run_001/onestop/scores.json",
        }
    ]

    csv_output = result_aggregation.format_csv(rows)
    json_output = result_aggregation.format_json(rows)

    assert csv_output.splitlines()[0].startswith("run,pipeline,model,checkpoint,sari")
    assert "run_001,onestop,google/flan-t5-base,,41.5000" in csv_output
    assert '"sari": 41.5' in json_output
    assert '"fkgl": null' in json_output

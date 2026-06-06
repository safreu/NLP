from pathlib import Path

from evaluation import result_aggregation
from storage.json_store import write_json


def write_score_file(path: Path, data: object) -> None:
    write_json(data, path)


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
    scores_path = tmp_path / "run_001" / "onestop" / "scores.json"
    write_score_file(scores_path, metric_payload())

    rows = result_aggregation.aggregate_results([tmp_path / "run_001"])

    assert rows == [
        {
            "run": "run_001",
            "pipeline": "onestop",
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
    scores_path = tmp_path / "run_001" / "onestop" / "scores.json"
    write_score_file(scores_path, metric_payload())

    rows = result_aggregation.aggregate_results([scores_path])

    assert rows[0]["run"] == "run_001"
    assert rows[0]["pipeline"] == "onestop"


def test_aggregate_checkpoint_scores(tmp_path: Path) -> None:
    scores_path = tmp_path / "run_002" / "wikilarge" / "scores.json"
    write_score_file(
        scores_path,
        {
            "checkpoint-100": metric_payload(),
            "checkpoint-200": {"sari": 44.0, "bleu": 0.4},
        },
    )

    rows = result_aggregation.aggregate_results([tmp_path / "run_002"])

    assert [row["checkpoint"] for row in rows] == ["checkpoint-100", "checkpoint-200"]
    assert rows[0]["pipeline"] == "wikilarge"
    assert rows[0]["sari"] == 41.5
    assert rows[1]["sari"] == 44.0
    assert rows[1]["fkgl"] is None
    assert rows[1]["bleu"] == 0.4


def test_aggregate_flat_legacy_scores(tmp_path: Path) -> None:
    scores_path = tmp_path / "run_003" / "scores.json"
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

    rows = result_aggregation.aggregate_results([tmp_path / "run_003"])

    assert rows[0]["run"] == "run_003"
    assert rows[0]["pipeline"] == ""
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

    assert "| run | pipeline | checkpoint | sari | fkgl |" in markdown
    assert "| run_001 | onestop |  | 41.5000 | NA |" in markdown


def test_format_csv_and_json_outputs() -> None:
    rows: list[result_aggregation.ResultRow] = [
        {
            "run": "run_001",
            "pipeline": "onestop",
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

    assert csv_output.splitlines()[0].startswith("run,pipeline,checkpoint,sari")
    assert "run_001,onestop,,41.5000" in csv_output
    assert '"sari": 41.5' in json_output
    assert '"fkgl": null' in json_output

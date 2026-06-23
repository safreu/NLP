from pathlib import Path

from evaluation import result_aggregation, result_visualization
from storage.json_store import write_json


def sample_rows() -> list[result_aggregation.ResultRow]:
    return [
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
            "scores_path": "runs/run_001/onestop/scores.json",
        },
        {
            "run": "run_002",
            "pipeline": "wikilarge",
            "model": "google/flan-t5-base",
            "checkpoint": "",
            "sari": 44.0,
            "fkgl": 3.8,
            "bertscore_precision": 0.92,
            "bertscore_recall": 0.84,
            "bertscore_f1": 0.87,
            "bleu": 0.35,
            "rouge_l": 0.64,
            "token_f1": 0.77,
            "scores_path": "runs/run_002/wikilarge/scores.json",
        },
        {
            "run": "run_003",
            "pipeline": "wikilarge",
            "model": "facebook/bart-base",
            "checkpoint": "",
            "sari": 42.0,
            "fkgl": 4.5,
            "bertscore_precision": 0.89,
            "bertscore_recall": 0.8,
            "bertscore_f1": 0.84,
            "bleu": 0.31,
            "rouge_l": 0.6,
            "token_f1": 0.72,
            "scores_path": "runs/run_003/wikilarge/scores.json",
        },
    ]


def test_load_rows_from_csv_parses_numeric_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(result_aggregation.format_csv(sample_rows()) + "\n", encoding="utf-8")

    rows = result_visualization.load_rows_from_path(csv_path)

    assert rows[0]["model"] == "google/flan-t5-base"
    assert rows[0]["sari"] == 41.5
    assert rows[0]["bertscore_f1"] == 0.86


def test_visualize_results_writes_png_csv_and_markdown(tmp_path: Path) -> None:
    aggregated_path = tmp_path / "summary.json"
    output_dir = tmp_path / "visualizations"
    write_json(sample_rows(), aggregated_path)

    result_visualization.main([str(aggregated_path), "--output-dir", str(output_dir)])

    csv_path = output_dir / "readability_preservation_tradeoff.csv"
    markdown_path = output_dir / "readability_preservation_tradeoff.md"
    png_path = output_dir / "readability_preservation_tradeoff.png"

    assert csv_path.exists()
    assert markdown_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0
    assert "google/flan-t5-base" in csv_path.read_text(encoding="utf-8")
    assert "facebook/bart-base" in markdown_path.read_text(encoding="utf-8")

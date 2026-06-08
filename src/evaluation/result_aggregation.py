import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from storage.json_store import read_json

Number = int | float
MetricValue = str | Number | None
ResultRow = dict[str, MetricValue]

COLUMNS = [
    "run",
    "pipeline",
    "model",
    "checkpoint",
    "sari",
    "fkgl",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "bleu",
    "rouge_l",
    "token_f1",
    "scores_path",
]

METRIC_KEYS = {
    "asset_multi_reference_sari",
    "bert",
    "bertscore_f1_mean",
    "bertscore_precision_mean",
    "bertscore_recall_mean",
    "bleu",
    "f1",
    "flesch",
    "flesch_kincaid_mean",
    "metric",
    "rouge-l",
    "rouge_l",
    "sari",
    "token_f1_mean",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment scores.json files into report-ready tables."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories, parent directories containing runs, or scores.json files.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format. Markdown is the default because it can be pasted into reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Omit to print to stdout.",
    )
    parser.add_argument(
        "--missing-value",
        default="NA",
        help="Placeholder for missing metrics in CSV and Markdown output.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def normalize_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def load_scores(path: Path) -> dict[str, Any]:
    data = read_json(path)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return normalize_mapping(data)


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def as_number(value: Any) -> Number | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return value

    return None


def nested_number(data: Mapping[str, Any], *keys: str) -> Number | None:
    current: Any = data

    for key in keys:
        if not isinstance(current, Mapping):
            return None

        current = current.get(key)

    return as_number(current)


def nested_string(data: Mapping[str, Any], *keys: str) -> str | None:
    current: Any = data

    for key in keys:
        if not isinstance(current, Mapping):
            return None

        current = current.get(key)

    if isinstance(current, str):
        return current

    return None


def first_number(*values: Number | None) -> Number | None:
    for value in values:
        if value is not None:
            return value

    return None


def is_metric_payload(data: Mapping[str, Any]) -> bool:
    if data.get("metric") == "sari" and as_number(data.get("score")) is not None:
        return True

    return any(key in data for key in METRIC_KEYS)


def score_payloads(scores: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if is_metric_payload(scores):
        return [("", dict(scores))]

    payloads: list[tuple[str, dict[str, Any]]] = []
    for key, value in scores.items():
        if not isinstance(value, Mapping):
            continue

        payload = normalize_mapping(value)
        if is_metric_payload(payload):
            payloads.append((key, payload))

    return payloads


def extract_metrics(payload: Mapping[str, Any]) -> dict[str, Number | None]:
    sari = first_number(
        as_number(payload.get("sari")),
        as_number(payload.get("asset_multi_reference_sari")),
        as_number(payload.get("score")) if payload.get("metric") == "sari" else None,
    )

    return {
        "sari": sari,
        "fkgl": first_number(
            nested_number(payload, "flesch", "mean"),
            as_number(payload.get("flesch_kincaid_mean")),
            as_number(payload.get("fkgl")),
        ),
        "bertscore_precision": first_number(
            nested_number(payload, "bert", "precision_mean"),
            as_number(payload.get("bertscore_precision_mean")),
        ),
        "bertscore_recall": first_number(
            nested_number(payload, "bert", "recall_mean"),
            as_number(payload.get("bertscore_recall_mean")),
        ),
        "bertscore_f1": first_number(
            nested_number(payload, "bert", "f1_mean"),
            as_number(payload.get("bertscore_f1_mean")),
        ),
        "bleu": as_number(payload.get("bleu")),
        "rouge_l": first_number(
            as_number(payload.get("rouge-l")),
            as_number(payload.get("rouge_l")),
        ),
        "token_f1": first_number(
            nested_number(payload, "f1", "f1_mean"),
            as_number(payload.get("token_f1_mean")),
        ),
    }


def find_score_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.name == "scores.json" else []

    return sorted(path.rglob("scores.json"))


def infer_run_and_pipeline(root: Path, scores_path: Path) -> tuple[str, str]:
    if root.is_file():
        parent = scores_path.parent

        if parent.parent.name.startswith("run_"):
            return parent.parent.name, parent.name

        return parent.name, ""

    try:
        relative_parent = scores_path.parent.relative_to(root)
    except ValueError:
        return root.name, scores_path.parent.name

    parts = relative_parent.parts
    root_is_run = root.name.startswith("run_") or (root / "config.json").exists()

    if root_is_run:
        pipeline = parts[0] if parts else ""
        return root.name, pipeline

    run = parts[0] if parts else root.name
    pipeline = parts[1] if len(parts) > 1 else ""
    return run, pipeline


def find_run_root(scores_path: Path) -> Path | None:
    for parent in (scores_path.parent, *scores_path.parent.parents):
        if (parent / "config.json").exists():
            return parent

    return None


def load_model_lookup(run_root: Path) -> dict[str, str]:
    config_path = run_root / "config.json"
    if not config_path.exists():
        return {}

    data = read_json(config_path)
    if not isinstance(data, Mapping):
        return {}

    config = normalize_mapping(data)
    lookup: dict[str, str] = {}

    pipelines = config.get("pipelines")
    if isinstance(pipelines, Sequence) and not isinstance(pipelines, str | bytes | bytearray):
        for pipeline_config in pipelines:
            if not isinstance(pipeline_config, Mapping):
                continue

            pipeline_data = normalize_mapping(pipeline_config)
            pipeline_name = pipeline_data.get("name")
            if not isinstance(pipeline_name, str) or not pipeline_name:
                continue

            model_name = nested_string(pipeline_data, "training_config", "model_name")
            if model_name is not None:
                lookup[pipeline_name] = model_name

    datasets = config.get("datasets")
    baselines = config.get("baselines")
    if (
        isinstance(datasets, Sequence)
        and not isinstance(datasets, str | bytes | bytearray)
        and isinstance(baselines, Sequence)
        and not isinstance(baselines, str | bytes | bytearray)
    ):
        dataset_names = [item for item in datasets if isinstance(item, str) and item]
        baseline_names: list[str] = []

        for baseline_config in baselines:
            if not isinstance(baseline_config, Mapping):
                continue

            baseline_name = normalize_mapping(baseline_config).get("name")
            if isinstance(baseline_name, str) and baseline_name:
                baseline_names.append(baseline_name)

        for dataset_name in dataset_names:
            for baseline_name in baseline_names:
                lookup[f"{dataset_name}_{baseline_name}"] = baseline_name

    return lookup


def infer_model_name(
    pipeline: str,
    run_root: Path | None,
    model_lookup_cache: dict[Path, dict[str, str]],
) -> str:
    if run_root is None:
        return ""

    model_lookup = model_lookup_cache.get(run_root)
    if model_lookup is None:
        model_lookup = load_model_lookup(run_root)
        model_lookup_cache[run_root] = model_lookup

    if pipeline and pipeline in model_lookup:
        return model_lookup[pipeline]

    unique_models = set(model_lookup.values())
    if len(unique_models) == 1:
        return next(iter(unique_models))

    return ""


def row_for_payload(
    run: str,
    pipeline: str,
    model: str,
    checkpoint: str,
    scores_path: Path,
    payload: Mapping[str, Any],
) -> ResultRow:
    row: ResultRow = {
        "run": run,
        "pipeline": pipeline,
        "model": model,
        "checkpoint": checkpoint,
        "scores_path": str(scores_path),
    }
    row.update(extract_metrics(payload))
    return row


def aggregate_path(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    model_lookup_cache: dict[Path, dict[str, str]] = {}

    for scores_path in find_score_files(path):
        scores = load_scores(scores_path)
        run, pipeline = infer_run_and_pipeline(path, scores_path)
        model = infer_model_name(pipeline, find_run_root(scores_path), model_lookup_cache)

        for checkpoint, payload in score_payloads(scores):
            rows.append(row_for_payload(run, pipeline, model, checkpoint, scores_path, payload))

    return rows


def aggregate_results(paths: Sequence[Path]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    seen_score_files: set[Path] = set()

    for path in paths:
        for row in aggregate_path(path):
            scores_path = Path(str(row["scores_path"])).resolve()
            row_key = scores_path / str(row["checkpoint"])
            if row_key in seen_score_files:
                continue

            seen_score_files.add(row_key)
            rows.append(row)

    return rows


def format_metric_value(value: MetricValue, missing_value: str) -> str:
    if value is None:
        return missing_value

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def output_row(row: Mapping[str, MetricValue], missing_value: str) -> dict[str, str]:
    return {column: format_metric_value(row.get(column), missing_value) for column in COLUMNS}


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def format_markdown(rows: Sequence[ResultRow], missing_value: str = "NA") -> str:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]

    for row in rows:
        formatted_row = output_row(row, missing_value)
        lines.append(
            "| " + " | ".join(markdown_escape(formatted_row[column]) for column in COLUMNS) + " |"
        )

    return "\n".join(lines)


def format_csv(rows: Sequence[ResultRow], missing_value: str = "NA") -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()

    for row in rows:
        writer.writerow(output_row(row, missing_value))

    return output.getvalue().rstrip("\n")


def format_json(rows: Sequence[ResultRow]) -> str:
    return json.dumps(list(rows), indent=4, ensure_ascii=False)


def format_rows(rows: Sequence[ResultRow], output_format: str, missing_value: str) -> str:
    if output_format == "csv":
        return format_csv(rows, missing_value)

    if output_format == "json":
        return format_json(rows)

    return format_markdown(rows, missing_value)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = aggregate_results(args.paths)
    output = format_rows(rows, args.format, args.missing_value)

    if args.output is None:
        print(output)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

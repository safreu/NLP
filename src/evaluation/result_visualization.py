import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation import result_aggregation

OUTPUT_BASENAME = "readability_preservation_tradeoff"
NUMERIC_COLUMNS = {
    "sari",
    "fkgl",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "bleu",
    "rouge_l",
    "token_f1",
}
MISSING_VALUE_TOKENS = {"", "NA", "None", "null"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize readability and preservation metrics from aggregated experiment results."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=(
            "Aggregated CSV/JSON files from aggregate-results, or run directories and scores.json "
            "files to aggregate on the fly."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "visualizations",
        help="Directory where the PNG, CSV, and Markdown outputs should be written.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_metric_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return value

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if stripped in MISSING_VALUE_TOKENS:
        return None

    try:
        return float(stripped)
    except ValueError:
        return None


def parse_text_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def normalize_row(row: Mapping[str, Any]) -> result_aggregation.ResultRow:
    normalized: result_aggregation.ResultRow = {}

    for column in result_aggregation.COLUMNS:
        if column in NUMERIC_COLUMNS:
            normalized[column] = parse_metric_value(row.get(column))
            continue

        normalized[column] = parse_text_value(row.get(column))

    return normalized


def load_aggregated_csv(path: Path) -> list[result_aggregation.ResultRow]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [normalize_row(row) for row in reader]


def load_aggregated_json(path: Path) -> list[result_aggregation.ResultRow]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")

    rows: list[result_aggregation.ResultRow] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected each row in {path} to be a JSON object")

        rows.append(normalize_row(result_aggregation.normalize_mapping(item)))

    return rows


def load_rows_from_path(path: Path) -> list[result_aggregation.ResultRow]:
    if path.is_dir() or path.name == "scores.json":
        return result_aggregation.aggregate_results([path])

    if path.suffix.lower() == ".csv":
        return load_aggregated_csv(path)

    if path.suffix.lower() == ".json":
        return load_aggregated_json(path)

    raise ValueError(
        f"Unsupported input path {path}. Use aggregated CSV/JSON files, run directories, "
        "or scores.json files."
    )


def row_key(row: Mapping[str, result_aggregation.MetricValue]) -> tuple[str, str, str, str]:
    return (
        str(row.get("scores_path", "")),
        str(row.get("checkpoint", "")),
        str(row.get("pipeline", "")),
        str(row.get("run", "")),
    )


def model_label(row: Mapping[str, result_aggregation.MetricValue]) -> str:
    for key in ("model", "pipeline", "run", "scores_path"):
        value = parse_text_value(row.get(key))
        if value:
            return value

    return "unknown"


def point_label(row: Mapping[str, result_aggregation.MetricValue]) -> str:
    run = parse_text_value(row.get("run"))
    pipeline = parse_text_value(row.get("pipeline"))
    checkpoint = parse_text_value(row.get("checkpoint"))

    primary = pipeline or run
    if run and pipeline and pipeline != run:
        primary = f"{run}/{pipeline}"

    if primary and checkpoint:
        return f"{primary} / {checkpoint}"

    if primary:
        return primary

    if checkpoint:
        return checkpoint

    return model_label(row)


def numeric_metric(
    row: Mapping[str, result_aggregation.MetricValue],
    key: str,
) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    return None


def collect_rows(paths: Sequence[Path]) -> list[result_aggregation.ResultRow]:
    rows: list[result_aggregation.ResultRow] = []
    seen_rows: set[tuple[str, str, str, str]] = set()

    for path in paths:
        for row in load_rows_from_path(path):
            key = row_key(row)
            if key in seen_rows:
                continue

            seen_rows.add(key)
            rows.append(normalize_row(row))

    return sorted(
        rows,
        key=lambda row: (
            model_label(row),
            parse_text_value(row.get("pipeline")),
            parse_text_value(row.get("run")),
            parse_text_value(row.get("checkpoint")),
        ),
    )


def summarize_bertscore_by_model(
    rows: Sequence[result_aggregation.ResultRow],
) -> list[tuple[str, float]]:
    scores_by_model: dict[str, list[float]] = {}

    for row in rows:
        bertscore_f1 = numeric_metric(row, "bertscore_f1")
        if bertscore_f1 is None:
            continue

        label = model_label(row)
        scores_by_model.setdefault(label, []).append(bertscore_f1)

    summary = [
        (label, sum(scores) / len(scores)) for label, scores in scores_by_model.items() if scores
    ]

    return sorted(summary, key=lambda item: item[1], reverse=True)


def write_plot(rows: Sequence[result_aggregation.ResultRow], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    scatter_rows = [
        row
        for row in rows
        if numeric_metric(row, "sari") is not None and numeric_metric(row, "fkgl") is not None
    ]
    bertscore_summary = summarize_bertscore_by_model(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    scatter_ax, bar_ax = axes

    if scatter_rows:
        labels = sorted({model_label(row) for row in scatter_rows})
        color_map = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
        colors = {label: color_map(index % color_map.N) for index, label in enumerate(labels)}

        for label in labels:
            model_rows = [row for row in scatter_rows if model_label(row) == label]
            fkgl_values = [numeric_metric(row, "fkgl") for row in model_rows]
            sari_values = [numeric_metric(row, "sari") for row in model_rows]
            x_values = [value for value in fkgl_values if value is not None]
            y_values = [value for value in sari_values if value is not None]

            scatter_ax.scatter(
                x_values,
                y_values,
                color=colors[label],
                edgecolors="black",
                alpha=0.85,
                label=label,
                linewidths=0.5,
                s=75,
            )

            for row, x_value, y_value in zip(model_rows, x_values, y_values, strict=False):
                scatter_ax.annotate(
                    point_label(row),
                    (x_value, y_value),
                    fontsize=8,
                    textcoords="offset points",
                    xytext=(4, 4),
                )

        scatter_ax.set_title("SARI vs Flesch-Kincaid")
        scatter_ax.set_xlabel("Flesch-Kincaid grade level (lower is simpler)")
        scatter_ax.set_ylabel("SARI")
        scatter_ax.grid(True, alpha=0.3)
        scatter_ax.legend(fontsize=8, title="Model", title_fontsize=9)
    else:
        scatter_ax.text(0.5, 0.5, "No rows with both SARI and FKGL.", ha="center", va="center")
        scatter_ax.set_axis_off()

    if bertscore_summary:
        model_names = [item[0] for item in bertscore_summary]
        bertscore_values = [item[1] for item in bertscore_summary]

        bar_ax.barh(model_names, bertscore_values, color="#4e79a7")
        bar_ax.invert_yaxis()
        bar_ax.set_title("Mean BERTScore F1 by model")
        bar_ax.set_xlabel("BERTScore F1")
        bar_ax.set_xlim(0.0, max(1.0, max(bertscore_values) + 0.05))
        bar_ax.grid(True, axis="x", alpha=0.3)

        for index, value in enumerate(bertscore_values):
            bar_ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=8)
    else:
        bar_ax.text(0.5, 0.5, "No rows with BERTScore F1.", ha="center", va="center")
        bar_ax.set_axis_off()

    fig.suptitle("Readability-preservation trade-off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(
    rows: Sequence[result_aggregation.ResultRow],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{OUTPUT_BASENAME}.csv"
    markdown_path = output_dir / f"{OUTPUT_BASENAME}.md"
    plot_path = output_dir / f"{OUTPUT_BASENAME}.png"

    csv_path.write_text(result_aggregation.format_csv(rows) + "\n", encoding="utf-8")
    markdown_path.write_text(result_aggregation.format_markdown(rows) + "\n", encoding="utf-8")
    write_plot(rows, plot_path)

    return csv_path, markdown_path, plot_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = collect_rows(args.paths)
    if not rows:
        raise SystemExit("No aggregated rows found for visualization.")

    csv_path, markdown_path, plot_path = write_outputs(rows, args.output_dir)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {markdown_path}")
    print(f"Wrote PNG: {plot_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

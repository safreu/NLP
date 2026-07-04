"""Run classical replacement-source ablations with WikiLarge and SimplePPDB++."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ClassicalMLConfig
from data.dataset_loader import DatasetLoader, Pair
from data.wikilarge_loader import WikiLargeLoader
from data.wikismall_loader import WikiSmallLoader
from evaluation.classical_evaluate import evaluate_classical_model
from preprocessing.classical_training_data import to_classical_pairs
from preprocessing.simpleppdb import (
    SIMPLEPPDB_DATA_URL,
    SIMPLEPPDB_REPOSITORY_URL,
    download_simpleppdb,
)
from storage.json_store import read_json, write_json
from training.classical_trainer import train_classical_model

CLASSIFIERS = ("logistic_regression", "svm", "random_forest")
REPLACEMENT_SOURCES = ("wikilarge", "simpleppdb")
DEFAULT_OUTPUT_DIR = Path("results") / "simpleppdb_ablation"
DEFAULT_SIMPLEPPDB_PATH = Path("data") / "external" / "simpleppdbpp_xl.tsv.gz"


@dataclass(frozen=True)
class RunResult:
    dataset: str
    replacement_source: str
    model_type: str
    output_dir: Path
    validation_scores: dict[str, Any]
    test_scores: dict[str, Any]
    predictions_csv: Path
    preservation_scores: dict[str, Any] | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("wikilarge", "wikismall"), default="wikilarge")
    parser.add_argument(
        "--replacement-source",
        choices=(*REPLACEMENT_SOURCES, "both"),
        default="simpleppdb",
        help="Use 'both' to rerun the WikiLarge baseline and SimplePPDB ablation together.",
    )
    parser.add_argument(
        "--model-type",
        choices=(*CLASSIFIERS, "all"),
        default="all",
        help="Classifier to run. The default runs all three classical classifiers.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-train-samples", type=non_negative_int, default=10000)
    parser.add_argument("--max-eval-samples", type=non_negative_int, default=2000)
    parser.add_argument("--simpleppdb-path", type=Path, default=DEFAULT_SIMPLEPPDB_PATH)
    parser.add_argument("--simpleppdb-url", default=SIMPLEPPDB_DATA_URL)
    parser.add_argument("--simpleppdb-min-score", type=float, default=0.0)
    parser.add_argument("--simpleppdb-max-candidates-per-source", type=positive_int, default=5)
    parser.add_argument("--simpleppdb-rule-limit", type=non_negative_int, default=None)
    parser.add_argument(
        "--skip-simpleppdb-download",
        action="store_true",
        help="Require --simpleppdb-path to already exist instead of downloading it.",
    )
    parser.add_argument(
        "--skip-generation-metrics",
        action="store_true",
        help="Only compute classifier metrics and prediction counts.",
    )
    parser.add_argument(
        "--run-preservation",
        action="store_true",
        help="Also run number and entity preservation summaries for each prediction file.",
    )
    return parser.parse_args(argv)


def resolve_limit(value: int | None) -> int | None:
    return None if value == 0 else value


def build_loader(args: argparse.Namespace) -> DatasetLoader:
    max_train_samples = resolve_limit(args.max_train_samples)
    max_eval_samples = resolve_limit(args.max_eval_samples)
    if args.dataset == "wikilarge":
        return WikiLargeLoader(
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
    return WikiSmallLoader(
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def selected_sources(replacement_source: str) -> list[str]:
    if replacement_source == "both":
        return list(REPLACEMENT_SOURCES)
    return [replacement_source]


def selected_models(model_type: str) -> list[str]:
    if model_type == "all":
        return list(CLASSIFIERS)
    return [model_type]


def write_prediction_csv(json_path: Path, csv_path: Path) -> None:
    rows = read_json(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "candidate", "reference"])
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            flatten_metric(f"{prefix}_{key}", nested_value, output)
    else:
        output[prefix] = value


def summary_row(result: RunResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": result.dataset,
        "replacement_source": result.replacement_source,
        "model_type": result.model_type,
        "output_dir": str(result.output_dir),
        "predictions_csv": str(result.predictions_csv),
    }
    for split, scores in (
        ("validation", result.validation_scores),
        ("test", result.test_scores),
    ):
        for key, value in scores.items():
            flatten_metric(f"{split}_{key}", value, row)
    if result.preservation_scores:
        for key, value in result.preservation_scores.items():
            flatten_metric(f"preservation_{key}", value, row)
    return row


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_preservation_metrics(
    predictions_csv: Path,
    *,
    dataset_name: str,
    model_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    from evaluation import entity_preservation, number_preservation

    rows, source_column, output_column, _ = number_preservation.load_rows(predictions_csv)
    number_scores = number_preservation.evaluate_model(
        model_name,
        dataset_name,
        rows,
        source_column,
        output_column,
    )
    entity_summary_rows, entity_type_rows, entity_example_rows = entity_preservation.evaluate_all(
        input_path=predictions_csv,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    preservation_dir = output_dir / "preservation"
    write_json(number_scores, preservation_dir / "number_preservation.json")
    write_json(entity_summary_rows, preservation_dir / "entity_preservation_summary.json")
    write_json(entity_type_rows, preservation_dir / "entity_preservation_by_type.json")
    write_json(entity_example_rows, preservation_dir / "entity_preservation_examples.json")
    entity_summary = entity_summary_rows[0] if entity_summary_rows else {}
    return {
        "number": number_scores,
        "entity_summary": entity_summary,
    }


def write_handoff(results: list[RunResult], run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# SimplePPDB++ Classical Ablation Handoff",
        "",
        "## Dataset Reference",
        "",
        f"- Resource: SimplePPDB++ from {SIMPLEPPDB_REPOSITORY_URL}",
        f"- Download URL used by the pipeline: {args.simpleppdb_url}",
        "- Citation: Maddela, Mounica and Xu, Wei. 2018. "
        "A Word-Complexity Lexicon and A Neural Readability Ranking Model for "
        "Lexical Simplification. EMNLP.",
        "- SimplePPDB++ format: phrase1, phrase2, relative complexity score, PPDB score.",
        "- Positive relative complexity means phrase1 is more complex than phrase2; "
        "negative means phrase2 is more complex than phrase1.",
        "",
        "## Experiment Setup",
        "",
        f"- Dataset evaluated: {args.dataset}",
        f"- Max train samples: {resolve_limit(args.max_train_samples)}",
        f"- Max validation/test samples: {resolve_limit(args.max_eval_samples)}",
        f"- Replacement source setting: {args.replacement_source}",
        f"- Classifier setting: {args.model_type}",
        f"- SimplePPDB min score: {args.simpleppdb_min_score}",
        f"- SimplePPDB max candidates per source: {args.simpleppdb_max_candidates_per_source}",
        f"- Generation metrics computed: {not args.skip_generation_metrics}",
        f"- Preservation metrics computed: {args.run_preservation}",
        "",
        "## Output Files",
        "",
        f"- Summary JSON: `{run_dir / 'summary.json'}`",
        f"- Summary CSV: `{run_dir / 'summary.csv'}`",
        f"- Handoff document: `{run_dir / 'handoff_for_chatgpt.md'}`",
        "",
        "## Runs",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result.replacement_source} / {result.model_type}",
                "",
                f"- Output directory: `{result.output_dir}`",
                f"- Predictions CSV: `{result.predictions_csv}`",
                f"- Test SARI: {result.test_scores.get('sari', 'not computed')}",
                f"- Test BLEU: {result.test_scores.get('bleu', 'not computed')}",
                f"- Test ROUGE-L: {result.test_scores.get('rouge-l', 'not computed')}",
                "",
            ]
        )
    (run_dir / "handoff_for_chatgpt.md").write_text("\n".join(lines), encoding="utf-8")


def run_one(
    *,
    dataset_name: str,
    train: list[Pair],
    valid: list[Pair],
    test: list[Pair],
    replacement_source: str,
    model_type: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> RunResult:
    model_dir = run_dir / replacement_source / model_type / "model"
    output_dir = run_dir / replacement_source / model_type
    config = ClassicalMLConfig(
        model_type=model_type,
        replacement_source=replacement_source,
        external_replacement_path=str(args.simpleppdb_path)
        if replacement_source == "simpleppdb"
        else None,
        simpleppdb_min_score=args.simpleppdb_min_score,
        simpleppdb_max_candidates_per_source=args.simpleppdb_max_candidates_per_source,
        simpleppdb_rule_limit=resolve_limit(args.simpleppdb_rule_limit),
        max_train_samples=resolve_limit(args.max_train_samples),
        max_eval_samples=resolve_limit(args.max_eval_samples),
        compute_generation_metrics=not args.skip_generation_metrics,
    )
    artifacts = train_classical_model(
        train_pairs=train,
        valid_pairs=valid,
        path=model_dir,
        config=config,
    )
    validation_scores = evaluate_classical_model(
        test_pairs=valid,
        artifacts=artifacts,
        predictions_path=output_dir / "validation_predictions.json",
        config=config,
        extra_metrics=artifacts.validation_metrics,
    )
    test_scores = evaluate_classical_model(
        test_pairs=test,
        artifacts=artifacts,
        predictions_path=output_dir / "test_predictions.json",
        config=config,
    )
    predictions_csv = output_dir / "test_predictions.csv"
    write_prediction_csv(output_dir / "test_predictions.json", predictions_csv)

    preservation_scores = None
    if args.run_preservation:
        preservation_scores = run_preservation_metrics(
            predictions_csv,
            dataset_name=dataset_name,
            model_name=f"{replacement_source}_{model_type}",
            output_dir=output_dir,
        )

    write_json(
        {
            "validation": validation_scores,
            "test": test_scores,
            "preservation": preservation_scores,
        },
        output_dir / "scores.json",
    )
    return RunResult(
        dataset=dataset_name,
        replacement_source=replacement_source,
        model_type=model_type,
        output_dir=output_dir,
        validation_scores=validation_scores,
        test_scores=test_scores,
        predictions_csv=predictions_csv,
        preservation_scores=preservation_scores,
    )


def run_ablation(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if "simpleppdb" in selected_sources(args.replacement_source):
        if args.skip_simpleppdb_download and not args.simpleppdb_path.exists():
            raise FileNotFoundError(f"SimplePPDB++ file not found: {args.simpleppdb_path}")
        if not args.skip_simpleppdb_download:
            download_simpleppdb(args.simpleppdb_path, args.simpleppdb_url)

    loader = build_loader(args)
    train_pairs, valid_pairs, test_pairs = loader.load_pairs()
    train = to_classical_pairs(train_pairs)
    valid = to_classical_pairs(valid_pairs)
    test = to_classical_pairs(test_pairs)

    write_json(
        {
            "dataset": args.dataset,
            "replacement_source": args.replacement_source,
            "model_type": args.model_type,
            "max_train_samples": resolve_limit(args.max_train_samples),
            "max_eval_samples": resolve_limit(args.max_eval_samples),
            "simpleppdb_path": str(args.simpleppdb_path),
            "simpleppdb_url": args.simpleppdb_url,
        },
        run_dir / "config.json",
    )

    results: list[RunResult] = []
    for replacement_source in selected_sources(args.replacement_source):
        for model_type in selected_models(args.model_type):
            print(f"Running {args.dataset} / {replacement_source} / {model_type}")
            results.append(
                run_one(
                    dataset_name=args.dataset,
                    train=train,
                    valid=valid,
                    test=test,
                    replacement_source=replacement_source,
                    model_type=model_type,
                    run_dir=run_dir,
                    args=args,
                )
            )

    summary_rows = [summary_row(result) for result in results]
    write_json(summary_rows, run_dir / "summary.json")
    write_summary_csv(summary_rows, run_dir / "summary.csv")
    write_handoff(results, run_dir, args)
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_dir = run_ablation(parse_args(argv))
    print(f"Finished SimplePPDB ablation: {run_dir}")


if __name__ == "__main__":
    main()

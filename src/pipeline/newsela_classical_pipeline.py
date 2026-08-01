"""Train and evaluate classical simplification models on Newsela."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from config import SEED, ClassicalMLConfig
from data.newsela_loader import NewselaLoader
from pipeline.classical_ml_pipeline import ClassicalMLPipeline
from storage.json_store import write_json
from training.classical_model import MODEL_SPECS

DEFAULT_MODELS = ("logistic_regression", "svm", "random_forest")


def _sample_limit(value: int) -> int | None:
    return None if value == 0 else value


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train classical Newsela simplifiers with article-disjoint train, validation, "
            "and test splits, then evaluate generation and preservation metrics."
        )
    )
    parser.add_argument("--encrypted-cache", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="dotenv file containing NEWSELA_CACHE_KEY (the key is never written to outputs).",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--max-train-samples",
        type=_non_negative_int,
        default=0,
        help="Pair cap after splitting; 0 uses the full training split.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=_non_negative_int,
        default=0,
        help="Per-split pair cap after splitting; 0 uses full validation and test splits.",
    )
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument("--min-replacement-count", type=int, default=2)
    parser.add_argument(
        "--skip-generation-metrics",
        action="store_true",
        help="Fast smoke-test mode; full runs should omit this flag.",
    )
    parser.add_argument(
        "--random-forest-estimators",
        type=int,
        default=100,
        help="Number of trees for the random-forest run.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    loader = NewselaLoader(
        encrypted_cache_path=args.encrypted_cache,
        env_file=args.env_file,
        max_train_samples=_sample_limit(args.max_train_samples),
        max_eval_samples=_sample_limit(args.max_eval_samples),
        random_state=args.random_state,
    )

    # Load and validate once; every model receives these exact same in-memory splits.
    train, validation, test = loader.load_pairs()
    if not train or not validation or not test:
        raise RuntimeError("Newsela produced an empty train, validation, or test split.")
    write_json(loader.split_metadata, output_path / "split_manifest.json")

    run_config = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "newsela",
        "models": list(args.models),
        "random_state": args.random_state,
        "max_train_samples": _sample_limit(args.max_train_samples),
        "max_eval_samples": _sample_limit(args.max_eval_samples),
        "compute_generation_metrics": not args.skip_generation_metrics,
        "metric_suite": [
            "classifier_accuracy_precision_recall_f1",
            "bert_score",
            "bleu",
            "token_f1",
            "flesch_kincaid_grade",
            "sari",
            "rouge_l",
            "entity_preservation",
            "number_preservation",
        ],
    }
    write_json(run_config, output_path / "run_config.json")

    for model_type in args.models:
        classifier_parameters = None
        if model_type == "random_forest":
            classifier_parameters = {"n_estimators": args.random_forest_estimators}
        config = ClassicalMLConfig(
            model_type=model_type,
            random_state=args.random_state,
            min_replacement_count=args.min_replacement_count,
            classifier_parameters=classifier_parameters,
            compute_generation_metrics=not args.skip_generation_metrics,
        )
        write_json(asdict(config), output_path / f"{model_type}_config.json")
        ClassicalMLPipeline(
            name=model_type,
            dataset_loader=loader,
            config=config,
        ).run(output_path)

    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = run(args)
    print(f"Newsela classical experiment finished: {destination}")


if __name__ == "__main__":
    main()

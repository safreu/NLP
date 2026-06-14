import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from config import TrainingConfig
from data.dataset_loader import DatasetLoader
from data.newsela_loader import NewselaLoader
from data.onestop_loader import OneStopLoader
from data.wikilarge_loader import WikiLargeLoader
from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.diversity_analyzer import DiversityAnalyzer
from evaluation.analyzers.error_case_analyser import ErrorCaseAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.analyzers.length_analyzer import LengthAnalyzer
from pipeline.evaluation_pipeline import EvaluationMode, EvaluationPipeline
from pipeline.training_pipeline import TrainingPipeline
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.run_store import create_run_dir

DEFAULT_DATASET = "all"

DEFAULT_WIKILARGE_MAX_TRAIN_SAMPLES = 10000
DEFAULT_WIKILARGE_MAX_EVAL_SAMPLES = 2000

DEFAULT_NEWSELA_MAX_TRAIN_SAMPLES = 10000
DEFAULT_NEWSELA_MAX_EVAL_SAMPLES = 2000

DATASET_CHOICES = ("all", "onestop", "wikilarge", "newsela")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset_loader: DatasetLoader
    config: TrainingConfig
    evaluation_mode: EvaluationMode
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def non_negative_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")

    return parsed_value


def positive_int(value: str) -> int:
    parsed_value = non_negative_int(value)

    if parsed_value == 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")

    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run text simplification experiments.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET,
        help="Dataset pipeline to run. The default reproduces the previous main.py behavior.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Hugging Face model name or local model path used for training.",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=None,
        help="Training epochs. Omit to use each dataset's default.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="Per-device train and evaluation batch size. Omit to use each dataset's default.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=[mode.value for mode in EvaluationMode],
        default=EvaluationMode.CHECKPOINTS.value,
        help="Evaluate only the final model or evaluate saved checkpoints.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Run output directory. Omit to create the next runs/run_XXX directory.",
    )
    parser.add_argument(
        "--wikilarge-max-train-samples",
        type=non_negative_int,
        default=None,
        help="WikiLarge train sample cap. Use 0 for the full split.",
    )
    parser.add_argument(
        "--wikilarge-max-eval-samples",
        type=non_negative_int,
        default=None,
        help="WikiLarge validation/test sample cap. Use 0 for the full splits.",
    )
    parser.add_argument(
        "--newsela-max-train-samples",
        type=non_negative_int,
        default=None,
        help="Newsela train sample cap. Use 0 for the full split.",
    )
    parser.add_argument(
        "--newsela-max-eval-samples",
        type=non_negative_int,
        default=None,
        help="Newsela validation/test sample cap. Use 0 for the full splits.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def selected_dataset_names(dataset: str) -> list[str]:
    if dataset == "all":
        return ["onestop", "wikilarge", "newsela"]

    return [dataset]


def apply_training_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    model_name = args.model_name if args.model_name is not None else config.model_name
    epochs = args.epochs if args.epochs is not None else config.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size

    return replace(config, model_name=model_name, epochs=epochs, batch_size=batch_size)


def resolve_sample_limit(value: int | None, default: int) -> int | None:
    if value is None:
        return default

    if value == 0:
        return None

    return value


def build_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    evaluation_mode = EvaluationMode(args.evaluation_mode)
    experiments: list[ExperimentSpec] = []

    for dataset_name in selected_dataset_names(args.dataset):
        if dataset_name == "onestop":
            experiments.append(
                ExperimentSpec(
                    name="onestop",
                    dataset_loader=OneStopLoader(),
                    config=apply_training_overrides(TrainingConfig(), args),
                    evaluation_mode=evaluation_mode,
                )
            )
            continue

        if dataset_name == "newsela":
            max_train_samples = resolve_sample_limit(
                args.newsela_max_train_samples,
                DEFAULT_NEWSELA_MAX_TRAIN_SAMPLES,
            )

            max_eval_samples = resolve_sample_limit(
                args.newsela_max_eval_samples,
                DEFAULT_NEWSELA_MAX_EVAL_SAMPLES,
            )

            experiments.append(
                ExperimentSpec(
                    name="newsela",
                    dataset_loader=NewselaLoader(
                        max_train_samples=max_train_samples,
                        max_eval_samples=max_eval_samples,
                    ),
                    config=apply_training_overrides(TrainingConfig(), args),
                    evaluation_mode=evaluation_mode,
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples,
                )
            )
            continue

        max_train_samples = resolve_sample_limit(
            args.wikilarge_max_train_samples,
            DEFAULT_WIKILARGE_MAX_TRAIN_SAMPLES,
        )
        max_eval_samples = resolve_sample_limit(
            args.wikilarge_max_eval_samples,
            DEFAULT_WIKILARGE_MAX_EVAL_SAMPLES,
        )
        experiments.append(
            ExperimentSpec(
                name="wikilarge",
                dataset_loader=WikiLargeLoader(
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples,
                ),
                config=apply_training_overrides(
                    TrainingConfig(epochs=3, max_target_length=128),
                    args,
                ),
                evaluation_mode=evaluation_mode,
                max_train_samples=max_train_samples,
                max_eval_samples=max_eval_samples,
            )
        )

    return experiments


def resolve_run_dir(output_path: Path | None) -> RunPaths:
    if output_path is None:
        run_dir = RunPaths(create_run_dir())
        return run_dir

    run_dir = RunPaths(output_path)
    run_dir.root.mkdir(parents=True, exist_ok=True)
    return run_dir


def training_config_data(config: TrainingConfig) -> dict[str, object]:
    data = asdict(config)
    data["generation_config"] = config.generation_config
    return data


def experiment_config_data(experiment: ExperimentSpec) -> dict[str, object]:
    data: dict[str, object] = {
        "name": experiment.name,
        "evaluation_mode": experiment.evaluation_mode.value,
        "training_config": training_config_data(experiment.config),
    }

    if experiment.max_train_samples is not None or experiment.max_eval_samples is not None:
        data["loader_config"] = {
            "max_train_samples": experiment.max_train_samples,
            "max_eval_samples": experiment.max_eval_samples,
        }

    return data


def run_config_data(
    args: argparse.Namespace,
    run_dir: Path,
    experiments: Sequence[ExperimentSpec],
) -> dict[str, object]:
    return {
        "dataset": args.dataset,
        "output_path": str(run_dir),
        "pipelines": [experiment_config_data(experiment) for experiment in experiments],
    }


def write_run_config(
    args: argparse.Namespace,
    run_dir: RunPaths,
    experiments: Sequence[ExperimentSpec],
) -> None:
    write_json(run_config_data(args, run_dir.root, experiments), run_dir.config_path)


def run_experiments(args: argparse.Namespace) -> RunPaths:
    run_dir: RunPaths = resolve_run_dir(args.output_path)
    experiments = build_experiments(args)

    write_run_config(args, run_dir, experiments)

    for experiment in experiments:
        TrainingPipeline(
            name=experiment.name,
            dataset_loader=experiment.dataset_loader,
            config=experiment.config,
            run_paths=run_dir,
            evaluation_pipeline=EvaluationPipeline(
                config=experiment.config,
                run_paths=run_dir,
                mode=experiment.evaluation_mode,
                analyzers=[
                    CopyAnalyzer(threshold=0.95),
                    InformationLossAnalyzer(),
                    LengthAnalyzer(),
                    DiversityAnalyzer(),
                    ErrorCaseAnalyzer(),
                ],
            ),
        ).run()

    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_experiments(parse_args(argv))


if __name__ == "__main__":
    main()

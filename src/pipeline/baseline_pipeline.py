import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from data.dataset_loader import Pair
from data.onestop_loader import OneStopLoader
from data.wikilarge_loader import WikiLargeLoader
from evaluation.metrics_builder import compute_all_metrics
from preprocessing.cleaner import normalize_whitespace, remove_prompt
from storage.json_store import write_json
from storage.prediction_store import prediction_rows
from storage.run_store import create_run_dir

DATASET_CHOICES = ("all", "onestop", "wikilarge")
BASELINE_CHOICES = ("copy", "punctuation_split")
DEFAULT_DATASET = "all"
DEFAULT_MAX_EXAMPLES = 20


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    predict: Callable[[str], str]


def non_negative_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")

    return parsed_value


def source_text(text: str) -> str:
    return normalize_whitespace(remove_prompt(text))


def copy_baseline(text: str) -> str:
    return source_text(text)


def ensure_terminal_punctuation(text: str) -> str:
    if not text or text[-1] in ".!?":
        return text

    return f"{text}."


def punctuation_split_baseline(text: str) -> str:
    simplified = source_text(text)
    simplified = re.sub(r"\s*(?:;|:)\s*", ". ", simplified)
    simplified = re.sub(r"\s*(?:--|\u2013|\u2014)\s*", ". ", simplified)
    simplified = re.sub(
        r",\s+(and|but|which|who|while|because)\s+",
        r". \1 ",
        simplified,
        flags=re.IGNORECASE,
    )

    return ensure_terminal_punctuation(normalize_whitespace(simplified))


BASELINES = {
    "copy": BaselineSpec(
        name="copy",
        description="Copies the source sentence after removing any training prompt.",
        predict=copy_baseline,
    ),
    "punctuation_split": BaselineSpec(
        name="punctuation_split",
        description="Splits on simple punctuation and common clause boundaries.",
        predict=punctuation_split_baseline,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate simple text simplification baselines with the project metrics."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET,
        help="Dataset to evaluate. The default evaluates every supported dataset.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=BASELINE_CHOICES,
        dest="baselines",
        default=None,
        help="Baseline to evaluate. Repeat to select multiple baselines. Default: all baselines.",
    )
    parser.add_argument(
        "--max-examples",
        type=non_negative_int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Number of test examples per dataset. Use 0 for the full test split.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Run output directory. Omit to create the next runs/run_XXX directory.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def selected_dataset_names(dataset: str) -> list[str]:
    if dataset == "all":
        return ["onestop", "wikilarge"]

    return [dataset]


def selected_baselines(names: Sequence[str] | None) -> list[BaselineSpec]:
    if names is None:
        names = BASELINE_CHOICES

    return [BASELINES[name] for name in names]


def sample_limit(max_examples: int) -> int | None:
    if max_examples == 0:
        return None

    return max_examples


def limit_pairs(pairs: Sequence[Pair], max_examples: int) -> list[Pair]:
    if max_examples == 0:
        return list(pairs)

    return list(pairs[:max_examples])


def load_test_pairs(dataset_name: str, max_examples: int) -> list[Pair]:
    if dataset_name == "onestop":
        _, _, test_pairs = OneStopLoader().load_pairs()
        return limit_pairs(test_pairs, max_examples)

    _, _, test_pairs = WikiLargeLoader(
        max_train_samples=0,
        max_eval_samples=sample_limit(max_examples),
    ).load_pairs()
    return limit_pairs(test_pairs, max_examples)


def resolve_run_dir(output_path: Path | None) -> Path:
    if output_path is None:
        return create_run_dir()

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_run_config(
    run_dir: Path,
    dataset_names: Sequence[str],
    baseline_specs: Sequence[BaselineSpec],
    max_examples: int,
) -> None:
    write_json(
        {
            "datasets": list(dataset_names),
            "baselines": [
                {"name": baseline.name, "description": baseline.description}
                for baseline in baseline_specs
            ],
            "max_examples": sample_limit(max_examples),
        },
        run_dir / "config.json",
    )


def evaluate_baseline(
    dataset_name: str,
    baseline: BaselineSpec,
    test_pairs: Sequence[Pair],
    run_dir: Path,
) -> Path:
    pipeline_dir = run_dir / f"{dataset_name}_{baseline.name}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    sources = [source_text(source) for source, _ in test_pairs]
    references = [reference for _, reference in test_pairs]
    predictions = [baseline.predict(source) for source in sources]
    scores = compute_all_metrics(sources, predictions, references)  # type: ignore[no-untyped-call]

    prediction_data = prediction_rows(sources, predictions, references)  # type: ignore[no-untyped-call]
    write_json(prediction_data, pipeline_dir / "predictions.json")
    write_json(scores, pipeline_dir / "scores.json")

    return pipeline_dir


def run_baselines(args: argparse.Namespace) -> Path:
    run_dir = resolve_run_dir(args.output_path)
    dataset_names = selected_dataset_names(args.dataset)
    baseline_specs = selected_baselines(args.baselines)

    write_run_config(run_dir, dataset_names, baseline_specs, args.max_examples)

    for dataset_name in dataset_names:
        test_pairs = load_test_pairs(dataset_name, args.max_examples)
        for baseline in baseline_specs:
            pipeline_dir = evaluate_baseline(dataset_name, baseline, test_pairs, run_dir)
            print(f"Wrote {dataset_name} {baseline.name} baseline to: {pipeline_dir}")

    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_dir = run_baselines(parse_args(argv))
    print(f"Baseline run written to: {run_dir}")


if __name__ == "__main__":
    main()

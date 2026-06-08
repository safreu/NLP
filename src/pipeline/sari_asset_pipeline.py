from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TrainingConfig
from prompts import elementary_prompt, intermediate_prompt
from storage.paths import RunPaths

DATASET_NAME = "facebook/asset"
DATASET_CONFIG = "simplification"
DEFAULT_SPLIT = "validation"
ORIGINAL_COLUMN = "original"
SIMPLIFICATIONS_COLUMN = "simplifications"

DEFAULT_MAX_EXAMPLES = 20
DEFAULT_PROMPT_LEVEL = "elementary"
DEFAULT_PREDICTIONS_PATH = Path("results/asset_sari_predictions.json")
DEFAULT_SCORE_PATH = Path("results/asset_sari_score.json")
RUN_PATHS = RunPaths.for_runs_root()


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained text simplification model on ASSET with SARI."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path or Hugging Face model name to evaluate. If omitted, the script uses "
            "a model from runs/latest.txt when available, otherwise TrainingConfig.model_name."
        ),
    )
    parser.add_argument(
        "--pipeline-name",
        default=None,
        help=(
            "Optional pipeline subdirectory inside the latest run, for example wikilarge. "
            "Used only when --model-path is omitted."
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=("validation", "test"),
        help="ASSET split to evaluate.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Number of ASSET examples to evaluate. Use 0 for the full split.",
    )
    parser.add_argument(
        "--prompt-level",
        default=DEFAULT_PROMPT_LEVEL,
        choices=("elementary", "intermediate", "none"),
        help="Prompt style used before each ASSET source sentence.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu or cuda. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Where to write ASSET sources, predictions, and references.",
    )
    parser.add_argument(
        "--score-path",
        type=Path,
        default=DEFAULT_SCORE_PATH,
        help="Where to write the SARI score metadata.",
    )

    return parser.parse_args(args)


def resolve_model_path(
    model_path: str | None,
    config: TrainingConfig | None = None,
    pipeline_name: str | None = None,
) -> str:
    if model_path:
        return model_path

    config = config or TrainingConfig()
    latest_model_dirs = get_latest_run_model_dirs(pipeline_name)
    if len(latest_model_dirs) == 1:
        return str(latest_model_dirs[0])

    if len(latest_model_dirs) > 1:
        candidates = ", ".join(str(path) for path in latest_model_dirs)
        raise RuntimeError(
            "Multiple model directories were found in the latest run. "
            f"Pass --model-path or --pipeline-name explicitly. Candidates: {candidates}"
        )

    if pipeline_name:
        raise FileNotFoundError(
            f"No model directory found for pipeline {pipeline_name!r} in the latest run."
        )

    return config.model_name


def get_latest_run_dir() -> Path | None:
    latest_run_file = RUN_PATHS.latest_txt_path

    if not latest_run_file.exists():
        return None

    latest_run_path = latest_run_file.read_text(encoding="utf-8").strip()
    if not latest_run_path:
        return None

    return Path(latest_run_path)


def get_latest_run_model_dirs(pipeline_name: str | None = None) -> list[Path]:
    latest_run_dir = get_latest_run_dir()
    if latest_run_dir is None:
        return []

    if pipeline_name:
        model_dir = latest_run_dir / pipeline_name / "model"
        return [model_dir] if model_dir.exists() else []

    model_dirs: list[Path] = []
    legacy_model_dir = latest_run_dir / "model"
    if legacy_model_dir.exists():
        model_dirs.append(legacy_model_dir)

    model_dirs.extend(sorted(path for path in latest_run_dir.glob("*/model") if path.exists()))
    return model_dirs


def load_asset_examples(
    split: str = DEFAULT_SPLIT,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> tuple[list[str], list[list[str]]]:
    split_name = split if max_examples <= 0 else f"{split}[:{max_examples}]"
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split_name)

    sources: list[str] = []
    references: list[list[str]] = []

    for row in dataset:
        sources.append(str(row[ORIGINAL_COLUMN]))
        references.append([str(reference) for reference in row[SIMPLIFICATIONS_COLUMN]])

    return sources, references


def select_device(device: str | None = None) -> str:
    if device:
        return device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_seq2seq_model(model_path: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return model, tokenizer


def build_prompt(source: str, prompt_level: str) -> str:
    if prompt_level == "elementary":
        return str(elementary_prompt(source))

    if prompt_level == "intermediate":
        return str(intermediate_prompt(source))

    return source


def generate_prediction(
    source: str,
    model: Any,
    tokenizer: Any,
    device: str,
    config: TrainingConfig,
    prompt_level: str,
) -> str:
    input_text = build_prompt(source, prompt_level)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=config.max_input_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **config.generation_config,
        )

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    return str(prediction).strip()


def generate_predictions(
    sources: Sequence[str],
    model: Any,
    tokenizer: Any,
    device: str,
    config: TrainingConfig,
    prompt_level: str,
) -> list[str]:
    return [
        generate_prediction(source, model, tokenizer, device, config, prompt_level)
        for source in sources
    ]


def write_predictions(
    path: Path,
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source": source,
            "prediction": prediction,
            "references": list(reference_set),
        }
        for source, prediction, reference_set in zip(
            sources,
            predictions,
            references,
            strict=True,
        )
    ]

    path.write_text(json.dumps(rows, indent=4, ensure_ascii=False), encoding="utf-8")


def write_score(
    path: Path,
    sari_score: float,
    model_path: str,
    split: str,
    max_examples: int,
    prompt_level: str,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_data = {
        "metric": "sari",
        "score": sari_score,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split": split,
        "max_examples": None if max_examples <= 0 else max_examples,
        "model_path": model_path,
        "prompt_level": prompt_level,
        "configured_model_name": config.model_name,
        "generation_config": config.generation_config,
    }

    path.write_text(json.dumps(score_data, indent=4, ensure_ascii=False), encoding="utf-8")


def print_examples(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> None:
    for index, (source, prediction, reference_set) in enumerate(
        zip(sources, predictions, references, strict=True),
        start=1,
    ):
        print(f"\nExample {index}")
        print(f"Source: {source}")
        print(f"Prediction: {prediction}")
        print("References:")
        for reference_index, reference in enumerate(reference_set[:3], start=1):
            print(f"  {reference_index}. {reference}")


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    model_path = resolve_model_path(args.model_path, config, args.pipeline_name)
    device = select_device(args.device)

    print(f"Loading ASSET split: {args.split}")
    sources, references = load_asset_examples(args.split, args.max_examples)

    print(f"Loading model: {model_path}")
    model, tokenizer = load_seq2seq_model(model_path, device)

    print(f"Generating {len(sources)} predictions on {device}")
    predictions = generate_predictions(
        sources,
        model,
        tokenizer,
        device,
        config,
        args.prompt_level,
    )

    from metrics.metric_sari import compute_sari

    sari_score = compute_sari(sources, predictions, references)
    write_predictions(args.predictions_path, sources, predictions, references)
    write_score(
        args.score_path,
        sari_score,
        model_path,
        args.split,
        args.max_examples,
        args.prompt_level,
        config,
    )

    print_examples(sources[:5], predictions[:5], references[:5])
    print(f"\nSARI on {len(sources)} ASSET {args.split} examples: {sari_score:.2f}")
    print(f"Predictions written to: {args.predictions_path}")
    print(f"Score written to: {args.score_path}")


if __name__ == "__main__":
    main()

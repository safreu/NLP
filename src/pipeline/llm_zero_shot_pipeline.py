from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config import SEED
from pipeline.sari_asset_pipeline import (
    DATASET_CONFIG,
    DATASET_NAME,
    DEFAULT_SPLIT,
    load_asset_examples,
    select_device,
    write_predictions,
)
from prompts import ZERO_SHOT_SIMPLIFY_INSTRUCTION, zero_shot_simplify_messages

# Default model is the instruction-tuned Gemma-4 12B variant
DEFAULT_MODEL_NAME = "google/gemma-4-12b-it"
DEFAULT_REVISION: str | None = None
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_EXAMPLES = 20

DEFAULT_PREDICTIONS_PATH = Path("results/asset_llm_zero_shot_predictions.json")
DEFAULT_SCORE_PATH = Path("results/asset_llm_zero_shot_score.json")

# Deterministic decoding: greedy, no sampling. Logged into score.json.
GENERATION_CONFIG: dict[str, Any] = {
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "do_sample": False,
    "num_beams": 1,
}


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local zero-shot Gemma simplification baseline on ASSET. "
            "Use the 'generate' step to produce predictions and the 'score' step "
            "to compute SARI on the saved predictions without re-generating."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate simplifications with a local Gemma model and write predictions JSON.",
    )
    generate_parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id of an instruction-tuned, open-weights causal LM.",
    )
    generate_parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned model revision (commit hash or tag) for reproducible downloads.",
    )
    generate_parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=("validation", "test"),
        help="ASSET split to evaluate.",
    )
    generate_parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Number of ASSET examples to generate. Use 0 for the full split.",
    )
    generate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of newly generated tokens per example.",
    )
    generate_parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu or cuda. Auto-detected when omitted.",
    )
    generate_parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Where to write ASSET sources, predictions, and references.",
    )
    generate_parser.set_defaults(func=run_generate)

    score_parser = subparsers.add_parser(
        "score",
        help="Compute ASSET SARI on a saved predictions JSON file.",
    )
    score_parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Predictions JSON written by the generate step.",
    )
    score_parser.add_argument(
        "--score-path",
        type=Path,
        default=DEFAULT_SCORE_PATH,
        help="Where to write the SARI score metadata.",
    )
    score_parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model id recorded in the score metadata (the model used to generate).",
    )
    score_parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Model revision recorded in the score metadata.",
    )
    score_parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=("validation", "test"),
        help="ASSET split recorded in the score metadata.",
    )
    score_parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Example cap recorded in the score metadata. Use 0 for the full split.",
    )
    score_parser.set_defaults(func=run_score)

    return parser.parse_args(args)


def get_hf_token() -> str | None:
    """Read the (optional) Hugging Face download token from the environment.

    Gemma weights are license-gated, so a token is required to download them.
    The token is only used for the download and is never written to any output.
    """
    token = os.environ.get("HF_TOKEN")
    return token or None


def resolve_dtype(device: str) -> torch.dtype:
    # bfloat16 is well supported on A100 GPUs; fall back to float32 elsewhere.
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_causal_model(
    model_name: str,
    revision: str | None,
    device: str,
    hf_token: str | None = None,
) -> tuple[Any, Any]:
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name, revision=revision, token=hf_token)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        token=hf_token,
        dtype=resolve_dtype(device),
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    return model, tokenizer


def generate_prediction(
    source: str,
    model: Any,
    tokenizer: Any,
    device: str,
    generation_config: dict[str, Any],
) -> str:
    messages = zero_shot_simplify_messages(source)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    prompt_length = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        output = model.generate(**inputs, **generation_config)

    # Decode only the newly generated tokens, dropping the prompt prefix.
    new_tokens = output[0][prompt_length:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return str(prediction).strip()


def generate_predictions(
    sources: Sequence[str],
    model: Any,
    tokenizer: Any,
    device: str,
    generation_config: dict[str, Any],
) -> list[str]:
    return [
        generate_prediction(source, model, tokenizer, device, generation_config)
        for source in sources
    ]


def read_predictions(path: Path) -> tuple[list[str], list[str], list[list[str]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))

    sources: list[str] = []
    predictions: list[str] = []
    references: list[list[str]] = []
    for row in rows:
        sources.append(str(row["source"]))
        predictions.append(str(row["prediction"]))
        references.append([str(reference) for reference in row["references"]])

    return sources, predictions, references


def write_score(
    path: Path,
    sari_score: float,
    model_name: str,
    revision: str | None,
    split: str,
    max_examples: int,
    generation_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_data = {
        "metric": "sari",
        "score": sari_score,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split": split,
        "max_examples": None if max_examples <= 0 else max_examples,
        "model_name": model_name,
        "revision": revision,
        "prompt": ZERO_SHOT_SIMPLIFY_INSTRUCTION,
        "seed": SEED,
        "generation_config": generation_config,
    }

    path.write_text(json.dumps(score_data, indent=4, ensure_ascii=False), encoding="utf-8")


def run_generate(args: argparse.Namespace) -> Path:
    set_seed(SEED)
    predictions_path: Path = args.predictions_path
    device = select_device(args.device)

    print(f"Loading ASSET split: {args.split}")
    sources, references = load_asset_examples(args.split, args.max_examples)

    print(f"Loading model: {args.model_name} (revision={args.revision}) on {device}")
    model, tokenizer = load_causal_model(
        args.model_name,
        args.revision,
        device,
        hf_token=get_hf_token(),
    )

    generation_config = {**GENERATION_CONFIG, "max_new_tokens": args.max_new_tokens}

    print(f"Generating {len(sources)} predictions on {device}")
    predictions = generate_predictions(sources, model, tokenizer, device, generation_config)

    write_predictions(predictions_path, sources, predictions, references)
    print(f"Predictions written to: {predictions_path}")

    return predictions_path


def run_score(args: argparse.Namespace) -> float:
    from metrics.metric_sari import compute_sari

    print(f"Reading predictions from: {args.predictions_path}")
    sources, predictions, references = read_predictions(args.predictions_path)

    sari_score: float = compute_sari(sources, predictions, references)

    write_score(
        args.score_path,
        sari_score,
        args.model_name,
        args.revision,
        args.split,
        args.max_examples,
        GENERATION_CONFIG,
    )

    print(f"SARI on {len(sources)} ASSET {args.split} examples: {sari_score:.2f}")
    print(f"Score written to: {args.score_path}")

    return sari_score


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

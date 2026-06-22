
import os
from collections.abc import Sequence
from typing import Any
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import zero_shot_simplify_messages
from evaluation.metrics_builder import compute_all_metrics
from storage.json_store import write_json
from storage.prediction_store import prediction_rows
torch.set_num_threads(16)


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


def select_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        torch_dtype=resolve_dtype(device),
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
        tokenize=True,
        enable_thinking=False,
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
    
def evaluate_llm(
    test_pairs,
    model_name: str,
    revision: str | None,
    generation_config: dict[str, Any],
    predictions_path: Path,
    device: str | None = None
):
    resolved_device = select_device(device)
    
    model, tokenizer = load_causal_model(
        model_name=model_name,
        revision=revision,
        device=resolved_device,
        hf_token=get_hf_token(),
    )
    
    sources = [input_text for input_text, _ in test_pairs]
    references = [ref for _, ref in test_pairs]
    
    candidates = generate_predictions(
        sources=sources,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        generation_config=generation_config,
    )
    
    write_json(
        prediction_rows(sources, candidates, references),
        predictions_path
    )
    
    return compute_all_metrics(sources, candidates, references)
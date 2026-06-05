from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TrainingConfig
from evaluation.metrics_builder import compute_all_metrics
from preprocessing.cleaner import remove_prompt
from storage.json_store import write_json
from storage.prediction_store import prediction_rows

torch.set_num_threads(16)


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_batch(input_texts: list[str], model, tokenizer, device, config: TrainingConfig):
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        max_length=config.max_input_length,
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **config.generation_config,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def generate_predictions(
    test_pairs,
    model,
    tokenizer,
    device,
    config: TrainingConfig,
    batch_size=16,
):
    candidates = []
    references = []

    for i in range(0, len(test_pairs), batch_size):
        batch = test_pairs[i : i + batch_size]

        input_texts = [input_text for input_text, _ in batch]
        batch_refs = [ref for _, ref in batch]

        predictions = generate_batch(
            input_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
        )

        candidates.extend(predictions)
        references.extend(batch_refs)

    return candidates, references


def extract_sources(test_pairs):
    return [remove_prompt(input_text) for input_text, _ in test_pairs]


def evaluate_model(
    test_pairs,
    config: TrainingConfig,
    model_path: str,
    predictions_path: str = "results.json",
):
    model, tokenizer, device = load_model(model_path)

    candidates, references = generate_predictions(test_pairs, model, tokenizer, device, config)

    sources = extract_sources(test_pairs)

    write_json(prediction_rows(sources, candidates, references), predictions_path)

    return compute_all_metrics(sources, candidates, references)


def evaluate_checkpoints(test_pairs, run_model_dir: str, config: TrainingConfig):
    model_dir = Path(run_model_dir)

    checkpoints = sorted(model_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))

    all_results = {}

    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint}")

        prediction_path = checkpoint / "predictions.json"

        results = evaluate_model(
            test_pairs=test_pairs,
            model_path=str(checkpoint),
            predictions_path=str(prediction_path),
            config=config,
        )

        all_results[checkpoint.name] = results

    return all_results

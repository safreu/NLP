from pathlib import Path

from preprocessing.cleaner import remove_prompt
from evaluation.file_writer import write_predictions
from evaluation.metrics_builder import compute_all_metrics
from config import (
    MODEL_OUTPUT_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_BEAMS,
    LENGTH_PENALTY,
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
torch.set_num_threads(16)

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_prediction_as_batch(input_texts: list[str], model, tokenizer, device):
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            do_sample=False,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            encoder_no_repeat_ngram_size=0,
        )
        
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def generate_prediction(input_text: str, model, tokenizer, device):
    return generate_prediction_as_batch(
        [input_text],
        model,
        tokenizer,
        device
    )[0]
    

def generate_predictions(test_pairs, model, tokenizer, device, batch_size=16):
    candidates = []
    references = []

    for i in range(0, len(test_pairs), batch_size):
        batch = test_pairs[i:i + batch_size]
        
        input_texts = [input_text for input_text, _ in batch]
        batch_refs = [ref for _, ref in batch]
        
        predictions = generate_prediction_as_batch(
            input_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        
        candidates.extend(predictions)
        references.extend(batch_refs)
    
    return candidates, references


def extract_sources(test_pairs):
    return [remove_prompt(input_text) for input_text, _ in test_pairs]


def evaluate_model(test_pairs, model_path: str=MODEL_OUTPUT_DIR, predictions_path: str="results.json"):
    model, tokenizer, device = load_model(model_path)

    candidates, references = generate_predictions(test_pairs, model, tokenizer, device)
    
    sources = extract_sources(test_pairs)
    
    write_predictions(sources, candidates, references, predictions_path)
    
    return compute_all_metrics(sources, candidates, references)


def evaluate_checkpoint(test_pairs, run_model_dir: str):
    model_dir = Path(run_model_dir)
    
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )
    
    all_results = {}
    
    for checkpoint in checkpoints:
        print(f"Evaluating {checkpoint}")
        
        prediction_path = checkpoint / "predictions.json"
        
        results = evaluate_model(
            test_pairs=test_pairs,
            model_path=str(checkpoint),
            predictions_path=str(prediction_path)
        )
        
        all_results[checkpoint.name] = results
        
    return all_results
        
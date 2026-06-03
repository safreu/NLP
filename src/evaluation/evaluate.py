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

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, device

def generate_prediction(input_text: str, model, tokenizer, device):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            do_sample=False,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=3,
        )
        
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_predictions(test_pairs, model, tokenizer, device):
    candidates = []
    references = []
 
    for input_text, reference in test_pairs:
        prediction = generate_prediction(input_text, model, tokenizer, device)
        
        candidates.append(prediction)
        references.append(reference)
        
    return candidates, references

def extract_sources(test_pairs):
    return [remove_prompt(input_text) for input_text, _ in test_pairs]

def evaluate_model(test_pairs, model_path: str=MODEL_OUTPUT_DIR, predictions_path: str="results.json"):
    model, tokenizer, device = load_model(model_path)

    candidates, references = generate_predictions(test_pairs, model, tokenizer, device)
    
    sources = extract_sources(test_pairs)
    
    write_predictions(sources, candidates, references, predictions_path)
    
    return compute_all_metrics(sources, candidates, references)
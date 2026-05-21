from metrics.Bleu import compute_bleuscore
from metrics.rouge import compute_rougescore
from metrics.bertScore import compute_bertscore
from metrics.metric_sari import compute_sari
from metrics.f1 import compute_f1
from metrics.flesch_kincaid import compute_flesch_kincaid

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def evaluate_model(test_pairs, model_path: str = "models/text-simplifier/OneStop"):
    
    candidates = []
    references = []
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    for input_text, reference in test_pairs:
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=4
            )
        
        prediction = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        
        candidates.append(prediction)
        references.append(reference)
        
    
    sources = [pair[0]
               .removeprefix("simplify to elementary: ")
               .removeprefix("simplify to intermediate: ")               
               .strip() for pair in test_pairs]
    references = [pair[1] for pair in test_pairs]
    results = {
        "bert": compute_bertscore(candidates, references),
        "bleu": compute_bleuscore(candidates, references),
        "f1": compute_f1(candidates, references),
        "flesch": compute_flesch_kincaid(candidates, references),
        "sari": compute_sari(sources, candidates, references),
        "rouge-l": compute_rougescore(candidates, references)
    }
    return results
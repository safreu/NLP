from metrics.Bleu import compute_bleuscore
from transformers import pipeline
from metrics.rouge import compute_rougescore
from metrics.bertScore import compute_bertscore
from metrics.metric_sari import compute_sari
from metrics.f1 import compute_f1
from metrics.flesch_kincaid import compute_flesch_kincaid

def evaluate_model(test_pairs, model_path: str = "models/text-simplifier/OneStop"):
    
    candidates = []
    references = []
    
    simplifier = pipeline(
        "text2text-generation",
        model=model_path,
        tokenizer=model_path,
        device=0
    )
    
    
    
    for input_text, reference in test_pairs:
        prediction = simplifier(
            input_text,
            max_new_tokens=256,
            do_sample=False
        )[0]["generated_text"]
        
        candidates.append(prediction)
        references.append(reference)
        
    
    sources = [pair[0].removeprefix("simplify: ") for pair in test_pairs]
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
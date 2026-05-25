from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu

def compute_bleuscore(candidates: list[str], references: list[str]) -> float:
    
    candidate_tokens = [candidate.split() for candidate in candidates]
    reference_tokens = [[reference.split()] for reference in references]
    
    score = corpus_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    )
    
    return score

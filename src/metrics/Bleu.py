from ast import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu

def compute_bleuscore(candidate: str, reference: str) -> float:
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    score = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    )

    print(score)
    return score

def compute_bleuscore(candidates: list[str], references: list[str]) -> float:
    
    candidate_tokens = [candidate.split() for candidate in candidates]
    reference_tokens = [reference.split() for reference in references]
    
    score = corpus_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    )
    
    return score

standard = [
    "The cat sits on the mat.",
    "A quick brown fox jumps.",
    "Completely unrelated sentence about quantum physics.",
]
simple = [
    "A cat is sitting on the mat.",
    "The fast brown fox is jumping.",
    "The weather is nice today.",
]

for ref, can in zip(standard, simple):
    compute_bleuscore(ref, can)

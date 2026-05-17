from rouge_score import rouge_scorer

def apply_rouge(candidate: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores = scorer.score(reference, candidate)

    print("Precision=", scores["rougeL"].precision)
    print("Recall=", scores["rougeL"].recall)
    print("F1=", scores["rougeL"].fmeasure)
    
    return scores


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
    apply_rouge(ref, can)


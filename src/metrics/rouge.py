from rouge_score import rouge_scorer

def compute_rougescore(candidate: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores = scorer.score(reference, candidate)

    print("Precision=", scores["rougeL"].precision)
    print("Recall=", scores["rougeL"].recall)
    print("F1=", scores["rougeL"].fmeasure)
    
    return scores["rougeL"].fmeasure

def compute_rougescore(candiates: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, can)["rougeL"].fmeasure
        for can, ref in zip(candiates, references, strict=True)
    ]
    
    return sum(scores) / len(scores)


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
    compute_rougescore(ref, can)


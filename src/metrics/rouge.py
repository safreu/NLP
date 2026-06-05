from rouge_score import rouge_scorer


def compute_rougescore(candiates: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, can)["rougeL"].fmeasure
        for can, ref in zip(candiates, references, strict=True)
    ]
    
    return sum(scores) / len(scores)


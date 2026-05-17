from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def apply_bleu(candidate: str, reference: str) -> float:
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    score = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    )

    print(score)
    return score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def compute_bleuscore(candidates: list[str], references: list[str]) -> float:
    # Each candidate should have exactly one matching reference sentence.
    if len(candidates) != len(references):
        raise ValueError(
            f"Candidates length ({len(candidates)}) and references length "
            f"({len(references)}) do not match."
        )

    candidate_tokens = [candidate.split() for candidate in candidates]
    reference_tokens = [[reference.split()] for reference in references]

    score = corpus_bleu(
        reference_tokens, candidate_tokens, smoothing_function=SmoothingFunction().method1
    )

    return score

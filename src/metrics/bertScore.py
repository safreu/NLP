from collections.abc import Sequence
from typing import TypedDict

import torch
from bert_score import score as _bert_score


class BERTScoreResult(TypedDict):
    precision: list[float]
    recall: list[float]
    f1: list[float]
    precision_mean: float
    recall_mean: float
    f1_mean: float


def compute_bertscore(
    standard: Sequence[str],
    simple: Sequence[str],
    *,
    lang: str = "en",
    model_type: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    rescale_with_baseline: bool = False,
    verbose: bool = False,
) -> BERTScoreResult:
    
    if len(standard) != len(simple):
        raise ValueError(f"Standard length ({len(standard)}) and simple length ({len(simple)}) doesn't match.")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    precision, recall, f1 = _bert_score(
        cands=list(standard),
        refs=list(simple),
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        verbose=verbose,
    )

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "precision_mean": float(precision.mean()),
        "recall_mean": float(recall.mean()),
        "f1_mean": float(f1.mean()),
    }


if __name__ == "__main__":

    # placeholder example for testing
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

    result = compute_bertscore(standard, simple, lang="en", verbose=True)

    for i, (c, ref) in enumerate(zip(standard, simple, strict=True)):
        print(f"[{i}] cand={c!r}")
        print(f"    ref ={ref!r}")
        print(
            f"    Precision: {result['precision'][i]:.4f} "
            f"Recall: {result['recall'][i]:.4f} "
            f"F: {result['f1'][i]:.4f}"
        )

    print(
        f"\nMean Precision: {result['precision_mean']:.4f} "
        f"Mean Recall={result['recall_mean']:.4f} "
        f"Mean F1: {result['f1_mean']:.4f}"
    )

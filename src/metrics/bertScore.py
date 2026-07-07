import sys
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
        raise ValueError(
            f"Standard length ({len(standard)}) and simple length ({len(simple)}) doesn't match."
        )

    pairs = []
    skipped = []

    for i, (cand, ref) in enumerate(zip(standard, simple, strict=True)):
        cand = cand.strip()
        ref = ref.strip()

        if not cand or not ref:
            skipped.append((i, cand, ref))
            continue

        pairs.append((cand, ref))

    if skipped:
        print(f"skipped {len(skipped)} empty candidates/references in BERTScore", file=sys.stderr)
        for i, cand, ref in skipped[:10]:
            print(f"entry at {i} empty: cand={cand!r}, ref={ref!r}", file=sys.stderr)

    if not pairs:
        return {
            "precision": [],
            "recall": [],
            "f1": [],
            "precision_mean": 0.0,
            "recall_mean": 0.0,
            "f1_mean": 0.0,
        }

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    standard, simple = zip(*pairs, strict=True)

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

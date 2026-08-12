from metrics.bertScore import compute_bertscore
from metrics.Bleu import compute_bleuscore
from metrics.f1 import compute_f1
from metrics.flesch_kincaid import compute_flesch_kincaid
from metrics.metric_sari import compute_sari
from metrics.preservation import compute_preservation_metrics
from metrics.rouge import compute_rougescore


def compute_all_metrics(
    sources: list[str], candidates: list[str], references: list[str]
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "bert": compute_bertscore(candidates, references),
        "bleu": compute_bleuscore(candidates, references),
        "f1": compute_f1(candidates, references),
        "flesch": compute_flesch_kincaid(candidates, references),
        "sari": compute_sari(sources, candidates, references),
        "rouge-l": compute_rougescore(candidates, references),
    }
    metrics.update(compute_preservation_metrics(sources, candidates, references))
    return metrics

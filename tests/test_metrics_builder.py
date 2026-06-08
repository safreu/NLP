import pytest

from evaluation import metrics_builder


def test_compute_all_metrics_collects_every_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace each metric with a stub so the test does not load any models
    # and only checks how compute_all_metrics wires the results together.
    monkeypatch.setattr(metrics_builder, "compute_bertscore", lambda c, r: "bert-value")
    monkeypatch.setattr(metrics_builder, "compute_bleuscore", lambda c, r: "bleu-value")
    monkeypatch.setattr(metrics_builder, "compute_f1", lambda c, r: "f1-value")
    monkeypatch.setattr(metrics_builder, "compute_flesch_kincaid", lambda c, r: "flesch-value")
    monkeypatch.setattr(metrics_builder, "compute_sari", lambda s, c, r: "sari-value")
    monkeypatch.setattr(metrics_builder, "compute_rougescore", lambda c, r: "rouge-value")

    result = metrics_builder.compute_all_metrics(
        sources=["source"],
        candidates=["candidate"],
        references=["reference"],
    )

    assert result == {
        "bert": "bert-value",
        "bleu": "bleu-value",
        "f1": "f1-value",
        "flesch": "flesch-value",
        "sari": "sari-value",
        "rouge-l": "rouge-value",
    }


def test_compute_all_metrics_passes_arguments_to_each_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple] = {}

    def record(name: str):
        def stub(*args):
            calls[name] = args
            return name

        return stub

    monkeypatch.setattr(metrics_builder, "compute_bertscore", record("bert"))
    monkeypatch.setattr(metrics_builder, "compute_bleuscore", record("bleu"))
    monkeypatch.setattr(metrics_builder, "compute_f1", record("f1"))
    monkeypatch.setattr(metrics_builder, "compute_flesch_kincaid", record("flesch"))
    monkeypatch.setattr(metrics_builder, "compute_sari", record("sari"))
    monkeypatch.setattr(metrics_builder, "compute_rougescore", record("rouge"))

    sources = ["a complex sentence"]
    candidates = ["a simple sentence"]
    references = ["a reference sentence"]

    metrics_builder.compute_all_metrics(sources, candidates, references)

    # SARI is the only metric that also receives the source sentences.
    assert calls["sari"] == (sources, candidates, references)
    for name in ("bert", "bleu", "f1", "flesch", "rouge"):
        assert calls[name] == (candidates, references)

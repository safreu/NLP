from types import SimpleNamespace

from metrics import preservation


class StubNLP:
    def pipe(self, texts):
        for text in texts:
            entities = []
            if "Alice" in text:
                entities.append(SimpleNamespace(text="Alice", label_="PERSON"))
            if "Paris" in text:
                entities.append(SimpleNamespace(text="Paris", label_="GPE"))
            yield SimpleNamespace(ents=entities)


def test_number_preservation_counts_duplicates_and_normalizes_format() -> None:
    result = preservation.compute_number_preservation(
        ["It cost 1,000 dollars in 2020 and rose 5%."],
        ["It cost 1000 dollars and rose 5 % ."],
        ["In 2020, it cost 1,000 dollars and rose 5%."],
    )

    assert result["source_number_count"] == 3
    assert result["candidate_preserved_count"] == 2
    assert result["candidate_preservation_rate"] == 2 / 3
    assert result["reference_preservation_rate"] == 1.0


def test_entity_preservation_reports_exact_text_and_type(monkeypatch) -> None:
    monkeypatch.setattr(preservation, "_NER_PIPELINE", StubNLP())

    result = preservation.compute_entity_preservation(
        ["Alice traveled to Paris."],
        ["Alice traveled."],
        ["Alice went to Paris."],
    )

    assert result["source_entity_count"] == 2
    assert result["candidate_preserved_count"] == 1
    assert result["candidate_preservation_rate"] == 0.5
    assert result["reference_preservation_rate"] == 1.0
    assert result["by_type"]["PERSON"]["candidate_preservation_rate"] == 1.0
    assert result["by_type"]["GPE"]["candidate_preservation_rate"] == 0.0

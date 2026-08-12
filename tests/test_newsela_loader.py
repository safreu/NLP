from dataclasses import dataclass

from data import newsela_loader
from data.newsela_loader import NewselaLoader, split_newsela_document_ids


@dataclass
class StubEntry:
    doc_id: str


class StubCorpus:
    def __init__(self, pairs_by_document: dict[str, list[tuple[str, str]]]) -> None:
        self.pairs_by_document = pairs_by_document
        self.entries = [StubEntry(doc_id) for doc_id in pairs_by_document]

    def subset(self, doc_ids: set[str]):
        return StubCorpus(
            {doc_id: pairs for doc_id, pairs in self.pairs_by_document.items() if doc_id in doc_ids}
        )

    def as_training_pairs(self, add_prompt: bool = True) -> list[tuple[str, str]]:
        return [pair for pairs in self.pairs_by_document.values() for pair in pairs]


def test_document_split_is_deterministic_and_disjoint() -> None:
    document_ids = [f"doc-{index}" for index in range(20)]
    first = split_newsela_document_ids(document_ids, random_state=42)
    second = split_newsela_document_ids(list(reversed(document_ids)), random_state=42)

    assert first == second
    train, validation, test = first
    assert len(train) == 16
    assert len(validation) == 2
    assert len(test) == 2
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_loader_removes_cross_split_source_duplicates(monkeypatch) -> None:
    pairs_by_document = {
        f"doc-{index}": [(f"source-{index}", f"target-{index}")] for index in range(20)
    }
    train_ids, validation_ids, test_ids = split_newsela_document_ids(
        list(pairs_by_document), random_state=42
    )
    train_doc = sorted(train_ids)[0]
    validation_doc = sorted(validation_ids)[0]
    test_doc = sorted(test_ids)[0]
    pairs_by_document[train_doc].append(("duplicate", "train target"))
    pairs_by_document[validation_doc].append(("duplicate", "validation target"))
    pairs_by_document[test_doc].append(("duplicate", "test target"))
    corpus = StubCorpus(pairs_by_document)
    monkeypatch.setattr(
        newsela_loader.NewselaCorpus,
        "load_from_disk",
        lambda **kwargs: corpus,
    )

    loader = NewselaLoader(random_state=42)
    train, validation, test = loader.load_pairs()

    assert "duplicate" in {source for source, _ in train}
    assert "duplicate" not in {source for source, _ in validation}
    assert "duplicate" not in {source for source, _ in test}
    assert loader.split_metadata["document_overlap"] is False
    assert loader.split_metadata["cross_split_source_overlap"] is False
    assert loader.split_metadata["dropped_cross_split_source_duplicates"] == {
        "validation": 1,
        "test": 1,
    }

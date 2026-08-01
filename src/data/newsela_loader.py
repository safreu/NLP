from dataclasses import dataclass, field
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import SEED
from data.corpus.newsela_corpus import NewselaCorpus
from data.dataset_loader import Pair


def split_newsela_document_ids(
    document_ids: list[str],
    *,
    random_state: int = SEED,
) -> tuple[set[str], set[str], set[str]]:
    """Create deterministic 80/10/10 article-level splits."""
    unique_ids = sorted(set(document_ids))
    if len(unique_ids) < 5:
        raise ValueError("Newsela needs at least five documents for an 80/10/10 split.")

    train_ids, held_out_ids = train_test_split(
        unique_ids,
        test_size=0.2,
        random_state=random_state,
    )
    valid_ids, test_ids = train_test_split(
        held_out_ids,
        test_size=0.5,
        random_state=random_state,
    )
    return set(train_ids), set(valid_ids), set(test_ids)


@dataclass
class NewselaLoader:
    name = "newsela"
    encrypted_cache_path: str | Path | None = None
    env_file: str | Path | None = None
    plaintext_path: str | Path | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    random_state: int = SEED
    split_metadata: dict[str, object] = field(default_factory=dict, init=False)
    _cached_splits: tuple[list[Pair], list[Pair], list[Pair]] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        if self._cached_splits is not None:
            return self._cached_splits

        if self.plaintext_path is not None:
            corpus = NewselaCorpus.load_from_disk(
                path=self.plaintext_path,
                encrypted_cache_path=self.encrypted_cache_path,
                env_file=self.env_file,
            )
        else:
            corpus = NewselaCorpus.load_from_disk(
                encrypted_cache_path=self.encrypted_cache_path,
                env_file=self.env_file,
            )
        if any(entry.doc_id is None for entry in corpus.entries):
            raise ValueError("Every Newsela entry must have a doc_id for leakage-safe splitting.")

        document_ids = [entry.doc_id for entry in corpus.entries if entry.doc_id is not None]
        train_ids, valid_ids, test_ids = split_newsela_document_ids(
            document_ids,
            random_state=self.random_state,
        )
        train = corpus.subset(train_ids).as_training_pairs()
        valid = corpus.subset(valid_ids).as_training_pairs()
        test = corpus.subset(test_ids).as_training_pairs()

        if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
            raise RuntimeError("Newsela document splits overlap.")

        train_sources = {source for source, _ in train}
        validation_before = len(valid)
        valid = [pair for pair in valid if pair[0] not in train_sources]
        blocked_test_sources = train_sources | {source for source, _ in valid}
        test_before = len(test)
        test = [pair for pair in test if pair[0] not in blocked_test_sources]
        dropped_source_duplicates = {
            "validation": validation_before - len(valid),
            "test": test_before - len(test),
        }

        if self.max_train_samples is not None:
            train = train[: self.max_train_samples]

        if self.max_eval_samples is not None:
            valid = valid[: self.max_eval_samples]
            test = test[: self.max_eval_samples]

        self.split_metadata = {
            "split_strategy": "article_doc_id_80_10_10",
            "random_state": self.random_state,
            "documents": {
                "train": len(train_ids),
                "validation": len(valid_ids),
                "test": len(test_ids),
            },
            "pairs_after_filtering_and_caps": {
                "train": len(train),
                "validation": len(valid),
                "test": len(test),
            },
            "document_ids": {
                "train": sorted(train_ids),
                "validation": sorted(valid_ids),
                "test": sorted(test_ids),
            },
            "document_overlap": False,
            "cross_split_source_overlap": False,
            "dropped_cross_split_source_duplicates": dropped_source_duplicates,
        }
        self._cached_splits = train, valid, test
        return self._cached_splits

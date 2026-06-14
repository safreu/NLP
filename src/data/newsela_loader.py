from dataclasses import dataclass

from data.dataset_loader import Pair
from data.newsela_corpus import NewselaCorpus
from preprocessing.dataset_builder import split_pairs


@dataclass
class NewselaLoader:
    name = "newsela"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        corpus = NewselaCorpus.load_from_disk()
        pairs = corpus.as_training_pairs()

        train, valid, test = split_pairs(pairs)

        if self.max_train_samples is not None:
            train = train[: self.max_train_samples]

        if self.max_eval_samples is not None:
            valid = valid[: self.max_eval_samples]
            test = test[: self.max_eval_samples]

        return train, valid, test

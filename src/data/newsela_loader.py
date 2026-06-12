from data.dataset_loader import Pair
from data.newsela_corpus import NewselaCorpus
from preprocessing.dataset_builder import split_pairs


class NewselaLoader:
    name = "newsela"

    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        corpus = NewselaCorpus.load_from_disk()
        pairs = corpus.as_training_pairs()
        return split_pairs(pairs)

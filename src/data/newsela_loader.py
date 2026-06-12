from data.dataset_loader import Pair
from preprocessing.dataset_builder import split_pairs
from data.newsela_corpus import NewselaCorpus

class NewselaLoader:
    name = "newsela"

    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        corpus = NewselaCorpus.load_from_disk()
        pairs = corpus.as_training_pairs()
        return split_pairs(pairs)

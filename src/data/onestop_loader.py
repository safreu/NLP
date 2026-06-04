from data.dataset_loader import Pair
from data.one_stop_english_corpus import OneStopEnglish
from preprocessing.dataset_builder import split_pairs

class OneStopLoader:
    name = "onestop"
    
    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        corpus = OneStopEnglish.load_from_disk()
        pairs = corpus.as_training_pairs
        return split_pairs(pairs)
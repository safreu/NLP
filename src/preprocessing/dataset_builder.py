from config import SEED
from datasets import Dataset
from sklearn.model_selection import train_test_split

def split_pairs(pairs, test_size=0.2, random_state=SEED):
    train_pairs, to_split_pairs = train_test_split(
        pairs,
        test_size=test_size,
        random_state=random_state
    )

    valid_pairs, test_pairs = train_test_split(
        to_split_pairs,
        test_size=0.5,
        random_state=random_state
    )
    return train_pairs, valid_pairs, test_pairs

def to_dataset(pairs):
    return Dataset.from_dict({
        "input": [pair[0] for pair in pairs],
        "target": [pair[1] for pair in pairs]
    })
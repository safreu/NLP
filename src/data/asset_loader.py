from datasets import load_dataset

from data.dataset_loader import Pair
from prompts import simplify_prompt

class AssetLoader:
    
    def __init__(
        self, 
        split: str = "test", 
        max_eval_samples: int | None = None, 
    ):
        self.split = split
        self.max_eval_samples = max_eval_samples
        
    def load_examples(self, add_prompt: bool = False) -> tuple[list[str], list[list[str]]]:
        dataset = load_dataset(
            "facebook/asset",
            "simplification",
            split=self.split
        )
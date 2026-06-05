from typing import Protocol

Pair = tuple[str, str]

class DatasetLoader(Protocol):
    name: str
    
    def load_pairs(self) -> tuple[list[Pair], list[Pair], list[Pair]]:
        ...
        

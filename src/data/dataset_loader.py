from typing import Protocol

Pair = tuple[str, str]


class DatasetLoader(Protocol):
    name: str

    def load_pairs(self, add_prompt: bool = True) -> tuple[list[Pair], list[Pair], list[Pair]]: ...

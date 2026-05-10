import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass
class SimplePPDB:
    entries: list[SimplePPDBEntry]

    @classmethod
    def load_from_disk(cls, path: str = "data/simple-ppdb/SimplePPDB") -> Self:
        """
        load data from given path

        Args:
            Path to the file to load the data from

        Returns:
            Instance of SimplePPDB

        Throws:
            Runtime Exception, in case the formating of the lines
            in the file dont match the expectations
        """

        cache_file = Path(path).with_suffix(".pkl")

        entries: list[SimplePPDBEntry] = []

        if cache_file.exists():
            with open(cache_file, "rb") as cached_file:
                entries = pickle.load(cached_file)

            return cls(entries)

        with open(path, encoding="utf8") as file:
            for line in file:
                match = re.match(
                    r"^(\d+\.\d+)\s+(\d+\.\d+)\s+\[([^\]]+)\]\s+(.+?)(?:\t|\s{2,})(.+)$", line
                )
                if match is None:
                    raise RuntimeError(f"Invalid format: {line}")

                p_score, s_score, category, input, output = match.groups()

                entries.append(
                    SimplePPDBEntry(
                        paraphrase_score=float(p_score),
                        simplification_score=float(s_score),
                        syntactic_category=category,
                        input=input,
                        output=output,
                    )
                )

        with open(cache_file, "wb") as cached_file:
            pickle.dump(entries, cached_file)

        return cls(entries)


@dataclass
class SimplePPDBEntry:
    paraphrase_score: float
    simplification_score: float
    syntactic_category: str
    input: str
    output: str

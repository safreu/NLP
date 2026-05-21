import pickle
import re
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Self


def normalize(text: str) -> str:    
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

@dataclass
class OneStopEnglishEntry:
    elementary: str
    intermediate: str
    advanced: str
    source: str

@dataclass
class OneStopEnglish:
    entries: list[OneStopEnglishEntry]

    @classmethod
    def load_from_disk(cls, path: str = "data/OneStopEnglishCorpus/Texts-Together-OneCSVperFile") -> Self:
        folder = Path(path)

        cache_file = folder.with_suffix(".pkl")

        if cache_file.exists():
            with open(cache_file, "rb") as cached_file:
                entries = pickle.load(cached_file)

            return cls(entries)

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exists {folder}")

        files = sorted(folder.glob("*.csv"))

        if not files:
            raise FileNotFoundError(f"No CSV files in {folder}")

        entries: list[OneStopEnglishEntry] = []

        for file in files:
            try:
                table = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                table = pd.read_csv(file, encoding="cp1252")

            columns = {"Elementary", "Intermediate", "Advanced"}

            table.columns = table.columns.str.strip()

            if not columns.issubset(table.columns):
                raise RuntimeError(f"{file} is missing {columns - set(table.columns)}")            

            for _, row in table.iterrows():
                entries.append(
                    OneStopEnglishEntry(
                        elementary=str(normalize(row["Elementary"])),
                        intermediate=str(normalize(row["Intermediate"])),
                        advanced=str(normalize(row["Advanced"])),
                        source=file.name
                    )
                )

        with open(cache_file, "wb") as cached_file:
            pickle.dump(entries, cached_file)

        return cls(entries)
    
    def as_training_pairs(self) -> list[tuple[str, str]]:
        pairs = []

        for entry in self.entries:
            pairs.append((
                f"simplify to elementary: {entry.advanced}",
                entry.elementary
            ))

            pairs.append((
                f"simplify to intermediate: {entry.advanced}",
                entry.intermediate
            ))

            pairs.append((
                f"simplify to elementary: {entry.intermediate}",
                entry.elementary
            ))
            
        return pairs
import pickle
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from prompts import elementary_prompt, intermediate_prompt
from preprocessing.cleaner import clean_text

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
                        elementary=str(clean_text(row["Elementary"])),
                        intermediate=str(clean_text(row["Intermediate"])),
                        advanced=str(clean_text(row["Advanced"])),
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
                elementary_prompt(entry.advanced),
                entry.elementary
            ))

            pairs.append((
                intermediate_prompt(entry.advanced),
                entry.intermediate
            ))

            pairs.append((
                elementary_prompt(entry.intermediate),
                entry.elementary
            ))
            
        return pairs
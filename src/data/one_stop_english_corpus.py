import pickle
from evaluation.evaluate import remove_prompt
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from prompts import elementary_prompt, intermediate_prompt
from preprocessing.cleaner import clean_text, remove_prompt
from preprocessing.filter import text_similarity, length_ratio

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
        similiar = 0
        ratio = 0
        for entry in self.entries:
            candidate_pairs = [
                (
                    elementary_prompt(entry.advanced),
                    entry.elementary
                ),
                (
                intermediate_prompt(entry.advanced),
                entry.intermediate
                ),
                (
                elementary_prompt(entry.intermediate),
                entry.elementary
                )
            ]
            
            for source, target in candidate_pairs:
                cleaned = remove_prompt(source)

                if text_similarity(cleaned, target) > 0.95:
                    similiar += 1
                    continue
                
                if length_ratio(cleaned, target) < 0.4:
                    ratio += 1
                    continue
                
                pairs.append((source, target))
                
        print(f"{str(similiar)} pairs were skipped, because of similiarity")
        print(f"{str(ratio)} pairs were skipped, because of ratio")
        
        return pairs
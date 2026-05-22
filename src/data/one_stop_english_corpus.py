import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from prompts import elementary_prompt, intermediate_prompt
from preprocessing.cleaner import clean_text, remove_prompt
from preprocessing.filter import text_similarity, length_ratio

@dataclass
class OneStopEnglishEntry:
    source: str
    target: str
    level: str
    source_file: str

@dataclass
class OneStopEnglish:
    entries: list[OneStopEnglishEntry]
    
    @staticmethod
    def _load_pair_file(file: Path, level: str) -> list[OneStopEnglishEntry]:
        text = file.read_text(encoding="utf-8", errors="replace")
        
        blocks = text.split("*******")
        
        pairs: list[OneStopEnglishEntry] = []
        
        for block in blocks:
            lines = [
                clean_text(line)
                for line in block.splitlines()
                if clean_text(line)
            ]
            
            if len(lines) < 2:
                continue
            
            source = lines[0]
            target = lines[1]
            
            if not source or not target:
                continue
            
            pairs.append(
                OneStopEnglishEntry(
                    source=source,
                    target=target,
                    level=level,
                    source_file=file.name
                )
            )
            
        return pairs
        
    @classmethod
    def load_from_disk(cls, path: str="data/OneStopEnglishCorpus/Sentence-Aligned") -> Self:
        folder = Path(path)

        cache_file = folder.with_suffix(".pkl")

        if cache_file.exists():
            with open(cache_file, "rb") as cached_file:
                entries = pickle.load(cached_file)

            return cls(entries)

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exists {folder}")

        files = {
            "ADV-ELE.txt": "elementary",
            "ADV-INT.txt": "intermediate",
            "ELE-INT.txt": "elementary"
        }


        entries: list[OneStopEnglishEntry] = []

        for filename, level in files.items():
            file = folder / filename
            
            if not file.exists():
                raise FileNotFoundError(f"No {filename} in {folder}")
            
            loaded = cls._load_pair_file(file, level)
            entries.extend(loaded)

        with open(cache_file, "wb") as cached_file:
            pickle.dump(entries, cached_file)

        return cls(entries)
    
    
    def as_training_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []

        similar = 0
        ratio = 0
        duplicate = 0
        
        seen: set[tuple[str, str]] = set()
        
        for entry in self.entries:
            if entry.level == "elementary":
                source = elementary_prompt(entry.source)
            elif entry.level == "intermediate":
                source = intermediate_prompt(entry.source)
            else:
                raise RuntimeError(f"Unknown level: {entry.level}")
            
            target = entry.target
            cleaned_source = remove_prompt(source)
            

            if text_similarity(cleaned_source, target) > 0.95:
                similar += 1
                continue
                
            if length_ratio(cleaned_source, target) < 0.2:
                ratio += 1
                continue
                
            training_pair = (source, target)
            
            if training_pair in seen:
                duplicate += 1
                continue

            seen.add(training_pair)
            pairs.append(training_pair)
                
        print(f"{similar} pairs were skipped, because of similarity")
        print(f"{ratio} pairs were skipped, because of ratio")
        print(f"{duplicate} duplicate pairs skipped")
        
        return pairs
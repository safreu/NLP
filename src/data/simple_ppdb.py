from typing import Self
from dataclasses import dataclass
from pathlib import Path
import re
import pickle

@dataclass
class SimplePPDB:
    entries: list[SimplePPDBEntry]



    @classmethod
    def load_from_disk(cls, path: str='data/simple-ppdb/SimplePPDB') -> Self:
        '''
        load data from given path

        Args:
            Path to the file to load the data from

        Returns:
            Instance of SimplePPDB

        Throws:
            Runtime Exception, in case the formating of the lines in the file dont match the expectations
        '''
        
        cache_file = Path(path).with_suffix('.pkl')

        entries: list[SimplePPDBEntry] = []

        if cache_file.exists():
            with open(cache_file, 'rb') as cached_file:
                entries = pickle.load(cached_file)

            return cls(entries)


        with open(path, 'r', encoding='utf8') as file:
            
            for line in file:

                match = re.match(
                    r'^(\d+\.\d+)\s+(\d+\.\d+)\s+\[([^\]]+)\]\s+(.+?)(?:\t|\s{2,})(.+)$',
                    line
                )
                if match is None:
                    raise RuntimeError(f'Invalid format: {line}')

                paraphrase_score, simplification_score, syntactic_category, input, output = match.groups()

                entries.append(
                    SimplePPDBEntry(
                        paraphrase_score=float(paraphrase_score),
                        simplification_score=float(simplification_score),
                        syntactic_category=syntactic_category,
                        input=input,
                        output=output
                    )
                )

        with open(cache_file, 'wb') as cached_file:
            pickle.dump(entries, cached_file)

        return cls(entries)
    

@dataclass
class SimplePPDBEntry:
    paraphrase_score: float
    simplification_score: float
    syntactic_category: str
    input: str
    output: str
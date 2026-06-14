from pathlib import Path
from typing import TypedDict

from storage.json_store import read_json, write_json

class PredictionRow(TypedDict):
    source: str
    candidate: str
    reference: str

def prediction_rows(sources, candidates, references) -> list[PredictionRow]:
    return [
        {
            "source": source,
            "candidate": candidate,
            "reference": reference,
        }
        for source, candidate, reference in zip(sources, candidates, references, strict=True)
    ]


def write_predictions(sources, candidates, references, path: Path) -> None:
    write_json(prediction_rows(sources, candidates, references), path)
    
def read_predictions(path: Path | str) -> list[PredictionRow]:
    read_json(path)

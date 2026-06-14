from pathlib import Path
from typing import TypedDict, cast

from storage.json_store import read_json, write_json


class PredictionRow(TypedDict):
    source: str
    candidate: str
    reference: str


def prediction_rows(
    sources: list[str], candidates: list[str], references: list[str]
) -> list[PredictionRow]:
    return [
        {
            "source": source,
            "candidate": candidate,
            "reference": reference,
        }
        for source, candidate, reference in zip(sources, candidates, references, strict=True)
    ]


def write_predictions(
    sources: list[str], candidates: list[str], references: list[str], path: Path
) -> None:
    write_json(prediction_rows(sources, candidates, references), path)


def read_predictions(path: Path | str) -> list[PredictionRow]:
    return cast(list[PredictionRow], read_json(path))

from pathlib import Path
from storage.json_store import write_json

def prediction_rows(sources, candidates, references):
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
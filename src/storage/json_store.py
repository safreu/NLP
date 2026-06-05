from pathlib import Path
import json
from typing import Any

def read_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    

def write_json(data: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data, 
            file, 
            indent=4, 
            ensure_ascii=False
        )
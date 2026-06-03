from pathlib import Path
import json
from typing import Any

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    

def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data, 
            file, 
            indent=4, 
            ensure_ascii=False
        )
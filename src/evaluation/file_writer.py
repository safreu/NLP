from pathlib import Path
import json

def write_results(results, path: str="results/evaluation_results.json"):
    folder = Path(path)
    folder.parent.mkdir(parents=True, exist_ok=True)
    
    with open(folder, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
    
    print(f"saved results to {folder}")
    
def write_predictions(sources, candidates, references, path: str="results/predictions.json"):
    folder = Path(path)
    folder.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [
        {
            "source": source,
            "candidate": candidate,
            "reference": reference,
        }
        for source, candidate, reference in zip(sources, candidates, references, strict=True)
    ]
    
    with open(folder, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4, ensure_ascii=False)
    
    print(f"saved predictions to {folder}")
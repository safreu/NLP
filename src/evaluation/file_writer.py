from pathlib import Path
import json

RUNS_DIR = Path("runs")

def create_run_dir() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    
    existing_runs = [
        int(path.name.replace("run_", ""))
        for path in RUNS_DIR.glob("run_*")
        if path.name.replace("run_", "").isdigit()
    ]
    
    next_run = max(existing_runs, default=0) + 1
    run_dir = RUNS_DIR / f"run_{next_run:03d}"
    run_dir.mkdir()
    
    (RUNS_DIR / "latest.txt").write_text(str(run_dir), encoding="utf-8")
    
    return run_dir    


def get_latest_run_dir() -> Path:
    latest_file = RUNS_DIR / "latest.txt"
    
    if not latest_file.exists():
        raise FileNotFoundError("No latest run file found")
    
    return Path(latest_file.read_text(encoding="utf-8").strip())


def get_run_dir(run_id: int) -> Path:
    return RUNS_DIR / f"run_{run_id:03d}"


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
    
def write_stats(stats, path: str="results/stats.json"):
    folder = Path(path)
    folder.parent.mkdir(parents=True, exist_ok=True)
    
    with open(folder, "w", encoding="utf-8") as file:
        json.dump(
            stats.to_dict(),
            file,
            indent=4,
            ensure_ascii=False
        )
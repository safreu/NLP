from pathlib import Path

RUNS_DIR = Path("runs")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

LATEST_RUN_FILE = RUNS_DIR / "latest.txt"

def get_run_dir(run_id: int) -> Path:
    return RUNS_DIR / f"run_{run_id:03d}"

def get_scores_path(run_dir: Path) -> Path:
    return run_dir / "scores.json"

def get_stats_path(run_dir: Path) -> Path:
    return run_dir / "stats.json"
from pathlib import Path

from storage.paths import LATEST_RUN_FILE, RUNS_DIR, get_run_dir


def create_run_dir() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    
    existing_runs = [
        int(path.name.replace("run_", ""))
        for path in RUNS_DIR.glob("run_*")
        if path.name.replace("run_", "").isdigit()
    ]
    
    next_run = max(existing_runs, default=0) + 1
    run_dir = get_run_dir(next_run)
    run_dir.mkdir()
    
    LATEST_RUN_FILE.write_text(str(run_dir), encoding="utf-8")
    
    return run_dir


def get_latest_run_dir() -> Path:
    if not LATEST_RUN_FILE.exists():
        raise FileNotFoundError("No latest run file found")
    
    return Path(LATEST_RUN_FILE.read_text(encoding="utf-8").strip())
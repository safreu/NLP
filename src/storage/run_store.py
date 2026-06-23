from pathlib import Path

from storage.paths import RunPaths


def create_run_dir(run_path: RunPaths | None = None) -> Path:
    if run_path is None:
        run_path = RunPaths.for_runs_root()

    run_path.root.mkdir(exist_ok=True)

    existing_runs = [
        int(path.name.replace("run_", ""))
        for path in run_path.root.glob("run_*")
        if path.name.replace("run_", "").isdigit()
    ]

    next_run = max(existing_runs, default=0) + 1
    run_dir = run_path.get_run_dir(next_run)
    run_dir.mkdir()

    run_path.latest_txt_path.write_text(str(run_dir), encoding="utf-8")

    return run_dir


def get_latest_run_dir(run_path: RunPaths | None = None) -> Path:
    if run_path is None:
        run_path = RunPaths.for_runs_root()

    if not run_path.latest_txt_path.exists():
        raise FileNotFoundError("No latest run file found")

    return Path(run_path.latest_txt_path.read_text(encoding="utf-8").strip())

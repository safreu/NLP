from storage.paths import RunPaths
from storage.run_store import create_run_dir


def test_create_run_dir_creates_next_run(tmp_path):
    run_paths = RunPaths.for_runs_root(tmp_path)

    first = create_run_dir(run_paths)
    second = create_run_dir(run_paths)

    assert first == tmp_path / "run_001"
    assert second == tmp_path / "run_002"
    assert run_paths.latest_txt_path.read_text(encoding="utf-8") == str(second)
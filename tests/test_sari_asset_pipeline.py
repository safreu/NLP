from pathlib import Path

import pytest

from config import TrainingConfig
from pipeline import sari_asset_pipeline
from storage.paths import RunPaths


def test_sari_asset_pipeline_imports() -> None:
    assert sari_asset_pipeline.DEFAULT_SPLIT == "validation"


def test_sari_asset_pipeline_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        sari_asset_pipeline.parse_args(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "Evaluate a trained text simplification model on ASSET with SARI." in output
    assert "--pipeline-name" in output


def test_resolve_model_path_prefers_explicit_model_path() -> None:
    model_path = sari_asset_pipeline.resolve_model_path("custom/model")

    assert model_path == "custom/model"


def test_resolve_model_path_falls_back_to_config_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sari_asset_pipeline,
        "RUN_PATHS",
        RunPaths(tmp_path),
    )
    config = TrainingConfig(model_name="example/base-model")

    model_path = sari_asset_pipeline.resolve_model_path(None, config)

    assert model_path == "example/base-model"


def test_resolve_model_path_uses_latest_pipeline_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run_001"
    model_dir = run_dir / "wikilarge" / "model"
    model_dir.mkdir(parents=True)
    run_paths = RunPaths(tmp_path / "runs")
    run_paths.latest_txt_path.write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(sari_asset_pipeline, "RUN_PATHS", run_paths)

    model_path = sari_asset_pipeline.resolve_model_path(None, pipeline_name="wikilarge")

    assert model_path == str(model_dir)


def test_resolve_model_path_rejects_ambiguous_latest_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run_001"
    (run_dir / "onestop" / "model").mkdir(parents=True)
    (run_dir / "wikilarge" / "model").mkdir(parents=True)
    run_paths = RunPaths(tmp_path / "runs")
    run_paths.latest_txt_path.write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(sari_asset_pipeline, "RUN_PATHS", run_paths)

    with pytest.raises(RuntimeError, match="Multiple model directories"):
        sari_asset_pipeline.resolve_model_path(None)

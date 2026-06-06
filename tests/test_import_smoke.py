import importlib
from importlib.metadata import entry_points

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "main",
        "evaluation.result_aggregation",
        "pipeline.training_pipeline",
        "pipeline.sari_asset_pipeline",
        "metrics.metric_sari",
        "storage.json_store",
        "storage.paths",
        "storage.prediction_store",
        "storage.run_store",
    ],
)
def test_core_modules_import_without_side_effects(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module is not None


def test_console_script_entry_point_loads() -> None:
    matches = [
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.name == "src"
    ]

    assert len(matches) == 1
    assert matches[0].value == "main:main"
    assert matches[0].load() is importlib.import_module("main").main


def test_metric_sari_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluate

    def fail_on_import_load(metric_name: str) -> None:
        raise AssertionError(f"Unexpected metric load during import: {metric_name}")

    module = importlib.import_module("metrics.metric_sari")
    with monkeypatch.context() as patch:
        patch.setattr(evaluate, "load", fail_on_import_load)
        module = importlib.reload(module)

    importlib.reload(module)

    assert module.get_sari_metric is not None

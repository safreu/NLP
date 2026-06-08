import importlib
from importlib.metadata import entry_points

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "main",
        "evaluation.result_aggregation",
        "evaluation.result_visualization",
        "pipeline.baseline_pipeline",
        "pipeline.training_pipeline",
        "pipeline.sari_asset_pipeline",
        "metrics.metric_sari",
        "metrics.Bleu",
        "metrics.rouge",
        "storage.json_store",
        "storage.paths",
        "storage.prediction_store",
        "storage.run_store",
    ],
)
def test_core_modules_import_without_side_effects(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    ("script_name", "entry_point_value"),
    [
        ("aggregate-results", "evaluation.result_aggregation:main"),
        ("evaluate-baselines", "pipeline.baseline_pipeline:main"),
        ("src", "main:main"),
        ("visualize-results", "evaluation.result_visualization:main"),
    ],
)
def test_console_script_entry_point_loads(script_name: str, entry_point_value: str) -> None:
    matches = [
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.name == script_name
    ]

    assert len(matches) == 1
    assert matches[0].value == entry_point_value

    module_name, function_name = entry_point_value.split(":", maxsplit=1)
    assert matches[0].load() is getattr(importlib.import_module(module_name), function_name)


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

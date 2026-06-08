from pathlib import Path

import main
from config import TrainingConfig
from pipeline.training_pipeline import EvaluationMode
from storage.json_store import read_json


def test_default_experiments_match_previous_main_behavior() -> None:
    experiments = main.build_experiments(main.parse_args([]))

    assert [experiment.name for experiment in experiments] == ["onestop", "wikilarge"]
    assert experiments[0].config == TrainingConfig()
    assert experiments[0].evaluation_mode is EvaluationMode.CHECKPOINTS
    assert experiments[1].config == TrainingConfig(epochs=3, max_target_length=128)
    assert experiments[1].evaluation_mode is EvaluationMode.CHECKPOINTS
    assert experiments[1].max_train_samples == 10000
    assert experiments[1].max_eval_samples == 2000


def test_cli_overrides_selected_experiment() -> None:
    experiments = main.build_experiments(
        main.parse_args(
            [
                "--dataset",
                "wikilarge",
                "--model-name",
                "example/model",
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--evaluation-mode",
                "final_model",
                "--wikilarge-max-train-samples",
                "50",
                "--wikilarge-max-eval-samples",
                "10",
            ]
        )
    )

    assert len(experiments) == 1
    assert experiments[0].name == "wikilarge"
    assert experiments[0].config.model_name == "example/model"
    assert experiments[0].config.epochs == 2
    assert experiments[0].config.batch_size == 4
    assert experiments[0].evaluation_mode is EvaluationMode.FINAL_MODEL
    assert experiments[0].max_train_samples == 50
    assert experiments[0].max_eval_samples == 10


def test_wikilarge_zero_sample_limit_means_full_split() -> None:
    experiments = main.build_experiments(
        main.parse_args(
            [
                "--dataset",
                "wikilarge",
                "--wikilarge-max-train-samples",
                "0",
                "--wikilarge-max-eval-samples",
                "0",
            ]
        )
    )

    assert experiments[0].max_train_samples is None
    assert experiments[0].max_eval_samples is None


def test_run_experiments_writes_config_and_runs_selected_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline_calls: list[tuple[str, Path, int]] = []

    class StubTrainingPipeline:
        def __init__(
            self,
            name,
            dataset_loader,
            config,
            run_paths,
            evaluation_mode,
        ):
            self.name = name
            self.config = config

        def run(self) -> None:
            pipeline_calls.append((self.name, output_path, self.config.epochs))

    monkeypatch.setattr(main, "TrainingPipeline", StubTrainingPipeline)

    output_path = tmp_path / "experiment"
    args = main.parse_args(
        [
            "--dataset",
            "wikilarge",
            "--epochs",
            "1",
            "--output-path",
            str(output_path),
        ]
    )

    run_dir = main.run_experiments(args)

    assert run_dir.root == output_path
    assert pipeline_calls == [("wikilarge", output_path, 1)]

    run_config = read_json(output_path / "config.json")
    assert run_config["dataset"] == "wikilarge"
    assert run_config["output_path"] == str(output_path)
    assert run_config["pipelines"][0]["name"] == "wikilarge"
    assert run_config["pipelines"][0]["evaluation_mode"] == "checkpoints"
    assert run_config["pipelines"][0]["training_config"]["epochs"] == 1
    assert run_config["pipelines"][0]["loader_config"] == {
        "max_train_samples": 10000,
        "max_eval_samples": 2000,
    }

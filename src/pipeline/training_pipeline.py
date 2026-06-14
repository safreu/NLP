from enum import Enum

from config import TrainingConfig
from data.dataset_loader import DatasetLoader
from evaluation.checkpoint_compare import compare_best_checkpoints
from evaluation.evaluate import evaluate_checkpoints, evaluate_model
from evaluation.prediction_analysis import analyze_prediction_copies
from preprocessing.dataset_builder import to_dataset
from storage.json_store import write_json
from storage.paths import RunPaths
from training.trainer import train_model


class EvaluationMode(Enum):
    FINAL_MODEL = "final_model"
    CHECKPOINTS = "checkpoints"


class TrainingPipeline:
    def __init__(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        config: TrainingConfig,
        run_paths: RunPaths,
        evaluation_mode: EvaluationMode = EvaluationMode.FINAL_MODEL,
    ):
        self.name = name
        self.dataset_loader = dataset_loader
        self.config = config
        self.run_paths = run_paths
        self.evaluation_mode = evaluation_mode

    def _evaluate(self, test):
        predictions_path = self.run_paths.predictions_path
        scores_path = self.run_paths.scores_path
        model_dir = self.run_paths.model_dir

        if self.evaluation_mode == EvaluationMode.CHECKPOINTS:
            results = evaluate_checkpoints(test, self.run_paths, self.config)

            write_json(results, scores_path)

            compare_best_checkpoints(
                scores_path=scores_path,
                model_dir=model_dir,
                output_path=self.run_paths.best_checkpoints_comparison_path,
                metric_path="sari",
                k=5,
                higher_is_better=True,
                copy_tresshold=0.95,
            )

            return

        results = evaluate_model(
            test_pairs=test,
            config=self.config,
            model_path=self.run_paths.model_path,
            predictions_path=self.run_paths.predictions_path,
        )

        write_json(results, scores_path)

        analyze_prediction_copies(
            predictions_path=predictions_path,
            output_path=self.run_paths.copy_analysis_path,
            copy_tresshold=0.95,
        )

    def run(self) -> None:
        self.run_paths.pipeline_dir = self.name

        print(f"Running Pipeline {self.name}")

        train, valid, test = self.dataset_loader.load_pairs()

        train_dataset = to_dataset(train)
        valid_dataset = to_dataset(valid)

        train_model(
            train=train_dataset,
            valid=valid_dataset,
            path=self.run_paths.model_dir,
            config=self.config,
        )

        self._evaluate(test)

        print(f"Finished Pipeline {self.name}")

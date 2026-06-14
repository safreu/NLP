
from enum import Enum

from config import TrainingConfig
from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.checkpoint_compare import compare_best_checkpoints
from evaluation.evaluate import evaluate_checkpoints, evaluate_model
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import read_predictions


class EvaluationMode(Enum):
    FINAL_MODEL = "final_model"
    CHECKPOINTS = "checkpoints"

class EvaluationPipeline:
    def __init__(
        self, 
        config: TrainingConfig, 
        run_paths: RunPaths,
        mode: EvaluationMode = EvaluationMode.FINAL_MODEL,
        analyzers: list | None = None
    ):
        self.config = config
        self.run_paths = run_paths
        self.mode = mode
        self.analyzers = analyzers or [
            CopyAnalyzer(),
            InformationLossAnalyzer()
        ]
       
        
    def _evaluate_checkpoints(self, test_pairs):
        results = evaluate_checkpoints(test_pairs, self.run_paths, self.config)
        
        write_json(results, self.run_paths.scores_path)

        compare_best_checkpoints(
            scores_path=self.run_paths.scores_path,
            model_dir=self.run_paths.model_dir,
            output_path=self.run_paths.best_checkpoints_comparison_path,
            metric_path="sari",
            k=5,
            higher_is_better=True,
            copy_tresshold=self.copy_treshold,
        )

        
    def _evaluate_final_model(self, test_pairs):
        results = evaluate_model(
            test_pairs=test_pairs,
            config=self.config,
            model_path=self.run_paths.model_path,
            predictions_path=self.run_paths.predictions_path,
        )

        write_json(results, self.run_paths.scores_path)
        
        predictions = read_predictions(self.run_paths.predictions_path)
        
        for analyzer in self.analyzers:
            analyzer.run(predictions, self.run_paths) 
        
    def run(self, test_pairs):
        if self.mode == EvaluationMode.CHECKPOINTS:
            self._evaluate_checkpoints(test_pairs)
        else: 
            self._evaluate_final_model(test_pairs)
        
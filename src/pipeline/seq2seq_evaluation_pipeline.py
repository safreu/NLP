from enum import Enum
from pathlib import Path

from configuration.seq2seq_config import GenerationConfig
from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.checkpoint_compare import compare_best_checkpoints
from evaluation.evaluate import evaluate_checkpoints, evaluate_model
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow, read_predictions


class EvaluationMode(Enum):
    FINAL_MODEL = "final_model"
    CHECKPOINTS = "checkpoints"


class Seq2SeqEvaluationPipeline:
    def __init__(
        self,
        generation_configs: list[GenerationConfig],
        run_paths: RunPaths,
        mode: EvaluationMode = EvaluationMode.FINAL_MODEL,
        analyzers: list | None = None,
    ):
        self.generation_configs = generation_configs
        self.run_paths = run_paths
        self.mode = mode
        self.analyzers = analyzers or [CopyAnalyzer(), InformationLossAnalyzer()]

    def _run_analyzers(self, predictions_path: Path, run_paths: RunPaths) -> None:
        predictions: list[PredictionRow] = read_predictions(predictions_path)

        for analyzer in self.analyzers:
            analyzer.run(predictions, run_paths)

    def _evaluate_checkpoints(self, test_pairs):
        results = evaluate_checkpoints(test_pairs, self.run_paths, self.generation_configs)

        write_json(results, self.run_paths.scores_path)

        for checkpoint_name in results:
            predictions_path = self.run_paths.checkpoint_predictions_path(checkpoint_name)

            if not predictions_path.exists():
                continue

            checkpoint_run_paths = RunPaths(predictions_path.parent)

            self._run_analyzers(predictions_path, checkpoint_run_paths)

        compare_best_checkpoints(
            scores_path=self.run_paths.scores_path,
            model_dir=self.run_paths.model_dir,
            output_path=self.run_paths.best_checkpoints_comparison_path,
            metric_path="sari",
            k=5,
            higher_is_better=True,
            copy_tresshold=0.95,
        )

    def _evaluate_final_model(self, test_pairs):
        all_results = {}

        for gen_idx, gen_conf in enumerate(self.generation_configs):
            gen_dir = self.run_paths.pipeline_dir / f"gen{gen_idx}"
            gen_dir.mkdir(parents=True, exist_ok=True)

            gen_run_paths = RunPaths(gen_dir)

            gen_conf.save(gen_dir)

            results = evaluate_model(
                test_pairs=test_pairs,
                config=gen_conf,
                model_path=self.run_paths.model_path,
                predictions_path=gen_run_paths.predictions_path,
            )

            write_json(results, gen_run_paths.scores_path)

            self._run_analyzers(gen_run_paths.predictions_path, gen_run_paths)

            all_results[f"gen{gen_idx}"] = results

        write_json(all_results, self.run_paths.scores_path)

    def run(self, test_pairs):

        if self.mode == EvaluationMode.CHECKPOINTS:
            self._evaluate_checkpoints(test_pairs)
        else:
            self._evaluate_final_model(test_pairs)

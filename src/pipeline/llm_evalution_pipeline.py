from pathlib import Path

from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.llm_evaluate import evaluate_llm
from storage.json_store import write_json
from storage.prediction_store import PredictionRow, read_predictions


class LLMEvaluationPipeline:
    def __init__(
        self,
        model_config,
        generation_config,
        run_paths,
        analyzers=None,
    ):
        self.model_config = model_config
        self.generation_config = generation_config
        self.run_paths = run_paths
        self.analyzers = analyzers or [CopyAnalyzer(), InformationLossAnalyzer()]

    def _run_analyzers(self, predictions_path: Path) -> None:
        predictions: list[PredictionRow] = read_predictions(predictions_path)

        for analyzer in self.analyzers:
            analyzer.run(predictions, self.run_paths)

    def run(self, test_pairs):

        results = evaluate_llm(
            test_pairs=test_pairs,
            model_name=self.model_config.model_name,
            revision=self.model_config.revision,
            device=self.model_config.device,
            generation_config=self.generation_config.to_dict(),
            predictions_path=self.run_paths.predictions_path,
        )

        write_json(results, self.run_paths.scores_path)

        self._run_analyzers(self.run_paths.predictions_path)

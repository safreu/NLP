from pathlib import Path

from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.llm_evaluate import evaluate_llm
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow, read_predictions


class LLMEvaluationPipeline:
    def __init__(
        self,
        model_config,
        generation_configs,
        run_paths,
        analyzers=None,
    ):
        self.model_config = model_config
        self.generation_configs = generation_configs
        self.run_paths = run_paths
        self.analyzers = analyzers or [CopyAnalyzer(), InformationLossAnalyzer()]

    def _run_analyzers(self, predictions_path: Path, run_paths: RunPaths) -> None:
        predictions: list[PredictionRow] = read_predictions(predictions_path)

        for analyzer in self.analyzers:
            analyzer.run(predictions, run_paths)

    def run(self, test_pairs):
        all_results = {}
        
        for gen_idx, gen_conf in enumerate(self.generation_configs):
            gen_dir = self.run_paths.pipeline_dir / f"gen{gen_idx}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            
            gen_run_paths = RunPaths(gen_dir)
            
            gen_conf.save(gen_dir)
            
            results = evaluate_llm(
                test_pairs=test_pairs,
                model_name=self.model_config.model_name,
                revision=self.model_config.revision,
                device=self.model_config.device,
                generation_config=gen_conf.to_dict(),
                predictions_path=gen_run_paths.predictions_path,
            )

            write_json(results, gen_run_paths.scores_path)

            self._run_analyzers(gen_run_paths.predictions_path, gen_run_paths)
            
            all_results[f"gen_{gen_idx}"] = results
            
        write_json(all_results, self.run_paths.scores_path)

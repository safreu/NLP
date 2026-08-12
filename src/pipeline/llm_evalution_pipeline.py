from collections.abc import Callable
from pathlib import Path

from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.llm_evaluate import (
    evaluate_llm,
    generate_predictions,
    get_hf_token,
    load_causal_model,
    select_device,
)
from prompts import zero_shot_simplify_messages
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow, read_predictions


class LLMEvaluationPipeline:
    def __init__(
        self,
        model_config,
        generation_configs,
        run_paths,
        message_builder=zero_shot_simplify_messages,
        analyzers=None,
        extra_evaluators=None,
    ):
        self.model_config = model_config
        self.generation_configs = generation_configs
        self.run_paths = run_paths
        self.message_builder = message_builder
        self.analyzers = analyzers or [CopyAnalyzer(), InformationLossAnalyzer()]
        self.extra_evaluators = extra_evaluators or []

    def _build_predict_fn(self, gen_conf) -> Callable[[list[str]], list[str]]:
        resolved_device = select_device(self.model_config.device)

        model, tokenizer = load_causal_model(
            model_name=self.model_config.model_name,
            revision=self.model_config.revision,
            device=resolved_device,
            hf_token=get_hf_token(),
        )

        def predict_fn(sources: list[str]) -> list[str]:
            return generate_predictions(
                sources=sources,
                model=model,
                tokenizer=tokenizer,
                device=resolved_device,
                generation_config=gen_conf.to_dict(),
                message_builder=self.message_builder,
            )

        return predict_fn

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

            results = evaluate_llm(
                test_pairs=test_pairs,
                model_name=self.model_config.model_name,
                revision=self.model_config.revision,
                device=self.model_config.device,
                generation_config=gen_conf.to_dict(),
                predictions_path=gen_run_paths.predictions_path,
                message_builder=self.message_builder,
            )

            predict_fn = self._build_predict_fn(gen_conf)

            for evaluator in self.extra_evaluators:
                extra_results = evaluator.run(
                    predict_fn=predict_fn,
                    output_dir=gen_dir,
                )
                results.update(extra_results)

            write_json(results, gen_run_paths.scores_path)

            self._run_analyzers(gen_run_paths.predictions_path, gen_run_paths)

            all_results[f"gen_{gen_idx}"] = results

        write_json(all_results, self.run_paths.scores_path)

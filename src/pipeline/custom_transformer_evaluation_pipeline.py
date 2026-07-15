from collections.abc import Callable
from typing import Any

from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.custom_transformer_evaluate import build_custom_transformer_predict_fn
from evaluation.metrics_builder import compute_all_metrics
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import prediction_rows, read_predictions


class CustomTransformerGenerationPipeline:
    def __init__(
        self,
        run_paths: RunPaths,
        max_length: int = 256,
        analyzers: list[Any] | None = None,
        extra_evaluators: list[Any] | None = None,
    ) -> None:
        self.run_paths = run_paths
        self.max_length = max_length
        self.analyzers = analyzers or [CopyAnalyzer(), InformationLossAnalyzer()]
        self.extra_evaluators = extra_evaluators or []

    def _run_anlyzers(self) -> None:
        predictions = read_predictions(self.run_paths.predictions_path)

        for analyzer in self.analyzers:
            analyzer.run(
                predictions,
                self.run_paths,
            )

    def _run_extra_evaluators(
        self,
        predict_fn: Callable[[list[str]], list[str]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for evaluator in self.extra_evaluators:
            evaluator_results = evaluator.run(
                predict_fn=predict_fn,
                output_dir=self.run_paths.output_dir,
            )
            results.update(evaluator_results)

        return results

    def run(self, model, tokenizer, device, test_pairs: list[tuple[str, str]]) -> None:
        sources = [source for source, _ in test_pairs]
        references = [reference for _, reference in test_pairs]

        predict_fn = build_custom_transformer_predict_fn(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=self.max_length,
        )

        candidates = predict_fn(sources)

        rows = prediction_rows(
            sources=sources,
            candidates=candidates,
            references=references,
        )

        write_json(rows, self.run_paths.predictions_path)

        results = compute_all_metrics(
            sources=sources,
            candidates=candidates,
            references=references,
        )

        extra_results = self._run_extra_evaluators(predict_fn)
        results.update(extra_results)

        write_json(results, self.run_paths.scores_path)

        self._run_anlyzers()

        return results

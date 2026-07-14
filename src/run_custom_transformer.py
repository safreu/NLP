import time
from pathlib import Path

from dotenv import load_dotenv

from configuration.custom_transformer_config import (
    CustomTransfomerGenerationConfig,
    CustomTransformerTrainingConfig,
)
from data.dataset_loader import DatasetLoader
from data.newsela_loader import NewselaLoader
from data.onestop_loader import OneStopLoader
from data.wikilarge_loader import WikiLargeLoader
from evaluation.analyzers.copy_analyzer import CopyAnalyzer
from evaluation.analyzers.diversity_analyzer import DiversityAnalyzer
from evaluation.analyzers.error_case_analyser import ErrorCaseAnalyzer
from evaluation.analyzers.information_loss_analyzer import InformationLossAnalyzer
from evaluation.analyzers.length_analyzer import LengthAnalyzer
from evaluation.analyzers.readability_analyzer import ReadabilityAnalyzer
from evaluation.asset_sari_evaluator import AssetSariEvaluator
from pipeline.custom_transformer_evaluation_pipeline import CustomTransformerGenerationPipeline
from storage.paths import RunPaths

load_dotenv()

def run_custom_transformer(
    trainings_configs: list[CustomTransformerTrainingConfig],
    generation_configs: list[CustomTransfomerGenerationConfig],
    dataset_loaders: list[DatasetLoader],
    run_dir: RunPaths,
):
    for dataset_loader in dataset_loaders:
        start = time.time()
        dataset_name = dataset_loader.__class__.__name__.replace("Loader", "")

        for train_idx, train_conf in enumerate(trainings_configs):
            run_name = f"{dataset_name}_train{train_idx}"

            CustomTransformerTrainingConfig(
                name=run_name,
                dataset_loader=dataset_loader,
                training_config=train_conf,
                run_paths=run_dir,
                evaluation_pipeline=CustomTransformerGenerationPipeline(
                    run_paths=run_dir,
                    analyzers=[
                        CopyAnalyzer(threshold=0.95),
                        InformationLossAnalyzer(),
                        LengthAnalyzer(),
                        DiversityAnalyzer(),
                        ErrorCaseAnalyzer(),
                        ReadabilityAnalyzer(),
                    ],
                    extra_evaluators=[AssetSariEvaluator(split="validation", max_examples=0)],
                ),
            ).run()

            print(
                f"{dataset_name} with {train_idx} finished in {(time.time()-start)/60:.1f} minutes"
            )


def main():

    training_configs = [
        CustomTransformerTrainingConfig(
            num_epochs=10,
            decoder_learning_rate=3e-4
        )
    ]
    

    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=10000, max_eval_samples=2000),
        WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        OneStopLoader(),
    ]

    run_dir = RunPaths.for_runs_root(Path("runs/seq2seq/finetune"))

    run_custom_transformer(training_configs, dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

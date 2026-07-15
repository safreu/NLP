import time
from pathlib import Path

from dotenv import load_dotenv

from configuration.custom_transformer_config import (
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
from models.custom_transformer.factory import TransformerVersion
from pipeline.custom_transformer_evaluation_pipeline import CustomTransformerGenerationPipeline
from pipeline.custom_transformer_training_pipeline import CustomTransformerTrainingPipeline
from storage.paths import RunPaths

load_dotenv()


def run_custom_transformer(
    trainings_configs: list[CustomTransformerTrainingConfig],
    dataset_loaders: list[DatasetLoader],
    run_dir: RunPaths,
):
    for dataset_loader in dataset_loaders:
        start = time.time()
        dataset_name = dataset_loader.__class__.__name__.replace("Loader", "")

        for train_idx, train_conf in enumerate(trainings_configs):
            run_name = f"{dataset_name}_{train_conf.version.value}_train{train_idx}"

            CustomTransformerTrainingPipeline(
                name=run_name,
                dataset_loader=dataset_loader,
                training_config=train_conf,
                run_paths=run_dir,
                evaluation_pipeline=CustomTransformerGenerationPipeline(
                    run_paths=run_dir,
                    max_length=train_conf.max_length,
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
                f"{dataset_name} with {train_idx} finished in ",
                f"{(time.time() - start) / 60:.1f} minutes",
            )


def main():

    training_configs = [
        CustomTransformerTrainingConfig(
            version=TransformerVersion.BASELINE,
            num_epochs=1,
            decoder_learning_rate=3e-4,
        ),
        CustomTransformerTrainingConfig(
            version=TransformerVersion.V1,
            num_epochs=1,
            decoder_learning_rate=3e-4,
        ),
        CustomTransformerTrainingConfig(
            version=TransformerVersion.V2,
            num_epochs=1,
            decoder_learning_rate=3e-4,
        ),
    ]

    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=10, max_eval_samples=2),
        WikiLargeLoader(max_train_samples=10, max_eval_samples=2),
        OneStopLoader(),
    ]

    run_dir = RunPaths.for_runs_root(Path("runs/custom_transformer"))

    run_custom_transformer(training_configs, dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

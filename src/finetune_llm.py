from pathlib import Path

from configuration.llm_config import (
    LLMGenerationConfig,
    LLMTrainingConfig,
    llm_generation_config_1_beam,
    llm_generation_config_1_contrastive,
    llm_generation_config_1_greedy,
    llm_generation_config_1_sampling,
    llm_training_config_1,
    llm_training_config_2,
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
from pipeline.llm_evalution_pipeline import LLMEvaluationPipeline
from pipeline.llm_training_pipeline import LLMTrainingPipeline
from storage.paths import RunPaths


def run_llm_finetune(
    trainings_configs, generation_configs, dataset_loaders: list[DatasetLoader], run_dir: RunPaths
):
    for dataset_idx, dataset_loader in enumerate(dataset_loaders):
        dataset_name = dataset_loader.__class__.__name__.replace("Loader", "")
        
        for train_idx, train_conf in enumerate(trainings_configs):
            LLMTrainingPipeline(
                name=f"LLM_{dataset_name}_train{train_idx}",
                dataset_loader=dataset_loader,
                training_config=train_conf,
                run_paths=run_dir,
                evaluation_pipeline=LLMEvaluationPipeline(
                    model_config=train_conf,
                    generation_config=generation_configs,
                    run_paths=run_dir,
                    analyzers=[
                        CopyAnalyzer(threshold=0.95),
                        InformationLossAnalyzer(),
                        LengthAnalyzer(),
                        DiversityAnalyzer(),
                        ErrorCaseAnalyzer(),
                        ReadabilityAnalyzer(),
                    ],
                    extra_evaluators=[
                        AssetSariEvaluator(
                            split="validation",
                            max_examples=0
                        )
                    ]
                ),
            ).run()


def run_llm_zeroshot(dataset_loaders: list[DatasetLoader], run_dir: RunPaths):

    for i, dataset_loader in enumerate(dataset_loaders):
        _, _, test = dataset_loader.load_pairs()

        run_dir.pipeline_dir = f"LLM_Zeroshot_{i}"

        LLMEvaluationPipeline(
            model_config=LLMTrainingConfig(),
            generation_config=LLMGenerationConfig(
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
            ),
            run_paths=run_dir,
            analyzers=[
                CopyAnalyzer(threshold=0.95),
                InformationLossAnalyzer(),
                LengthAnalyzer(),
                DiversityAnalyzer(),
                ErrorCaseAnalyzer(),
                ReadabilityAnalyzer(),
            ],
            extra_evaluators=[
                AssetSariEvaluator(
                    split="validation",
                    max_examples=0
                )
            ]
        ).run(test)


def main():
    
    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=10000, max_eval_samples=2000),
        WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        OneStopLoader(),
    ]

    run_dir = RunPaths.for_runs_root(Path("runs/llm/finetune"))
    #run_dir = RunPaths.for_runs_root(Path("runs/llm/zeroshot"))
    #run_dir = RunPaths.for_runs_root(Path("runs/llm/fewshot"))

    # run_llm_zeroshot(dataset_loaders, run_dir)

    trainings_configs: list[LLMTrainingConfig] = [
        llm_training_config_2,
        llm_training_config_1,    
    ]

    generation_configs: list[LLMGenerationConfig] = [
        llm_generation_config_1_greedy,
        llm_generation_config_1_beam,
        llm_generation_config_1_sampling,
        llm_generation_config_1_contrastive,
    ]

    run_llm_finetune(trainings_configs, generation_configs, dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

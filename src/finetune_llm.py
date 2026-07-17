import time
from pathlib import Path

from dotenv import load_dotenv

from configuration.llm_config import (
    LLMGenerationConfig,
    LLMTrainingConfig,
    llm_generation_config_1_beam,  # noqa: F401
    llm_generation_config_1_contrastive,  # noqa: F401
    llm_generation_config_1_greedy,  # noqa: F401
    llm_generation_config_1_sampling,  # noqa: F401
    llm_training_config_1,  # noqa: F401
    llm_training_config_2,  # noqa: F401
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

load_dotenv()


def run_llm_finetune(
    trainings_configs, generation_configs, dataset_loaders: list[DatasetLoader], run_dir: RunPaths
):
    for _, dataset_loader in enumerate(dataset_loaders):
        start = time.time()
        dataset_name = dataset_loader.__class__.__name__.replace("Loader", "")

        for train_idx, train_conf in enumerate(trainings_configs):
            LLMTrainingPipeline(
                name=f"LLM_{dataset_name}_train{train_idx}",
                dataset_loader=dataset_loader,
                training_config=train_conf,
                run_paths=run_dir,
                evaluation_pipeline=LLMEvaluationPipeline(
                    model_config=train_conf,
                    generation_configs=generation_configs,
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
                f"{dataset_name} with {train_idx} finished in ",
                f"{(time.time() - start) / 60:.1f} minutes",
            )


def run_llm_zeroshot(dataset_loaders: list[DatasetLoader], run_dir: RunPaths):

    for i, dataset_loader in enumerate(dataset_loaders):
        _, _, test = dataset_loader.load_pairs()

        run_dir.pipeline_dir = f"LLM_Zeroshot_{i}"

        LLMEvaluationPipeline(
            model_config=LLMTrainingConfig(),
            generation_configs=[
                LLMGenerationConfig(
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                )
            ],
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
        ).run(test)


def main():

    dataset_loaders: list[DatasetLoader] = [
        OneStopLoader(),
        WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        NewselaLoader(max_train_samples=10000, max_eval_samples=2000),
    ]

    run_dir = RunPaths.for_runs_root(Path("runs/llm/finetune"))
    # run_dir = RunPaths.for_runs_root(Path("runs/llm/zeroshot"))
    # run_dir = RunPaths.for_runs_root(Path("runs/llm/fewshot"))

    # run_llm_zeroshot(dataset_loaders, run_dir)

    # trainings_configs: list[LLMTrainingConfig] = [
    #    llm_training_config_2,
    #    llm_training_config_1,
    # ]

    # generation_configs: list[LLMGenerationConfig] = [
    #    llm_generation_config_1_greedy,
    #    llm_generation_config_1_beam,
    #    llm_generation_config_1_sampling,
    #    llm_generation_config_1_contrastive,
    # ]

    trainings_configs = [
        LLMTrainingConfig(
            model_name="google/gemma-4-E2B-it",
            use_qlora=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            learning_rate=2e-4,
            weight_decay=0.01,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=512,
            warmup_steps=100,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
        ),
    ]

    generation_configs = [
        LLMGenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            no_repeat_ngram_size=5,
        ),
        #LLMGenerationConfig(
        #    max_new_tokens=256,
        #    num_beams=4,
        #    early_stopping=True,
        #    no_repeat_ngram_size=5,
        #),
    ]

    run_llm_finetune(trainings_configs, generation_configs, dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

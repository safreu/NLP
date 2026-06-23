from configuration.llm_config import ZeroShotLLMConfig
from configuration.seq2seq_config import (
    GenerationConfig,
    TrainingConfig,
    generation_config_1,
    training_config_1,
    training_config_2,
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
from pipeline.seq2seq_evaluation_pipeline import Seq2SeqEvaluationPipeline
from pipeline.llm_evalution_pipeline import LLMEvaluationPipeline
from pipeline.seq2seq_training_pipeline import Seq2SeqTrainingPipeline
from storage.paths import RunPaths


def run_finetuning(traingings_configs, generation_configs, dataset_loaders, run_dir):
    for i in range(3):
        for j in range(3):
            Seq2SeqTrainingPipeline(
                name=f"Config_{i}{j}",
                dataset_loader=dataset_loaders[j],
                training_config=traingings_configs[i],
                run_paths=run_dir,
                evaluation_pipeline=Seq2SeqEvaluationPipeline(
                    generation_config=generation_configs[i],
                    run_paths=run_dir,
                    analyzers=[
                        CopyAnalyzer(threshold=0.95),
                        InformationLossAnalyzer(),
                        LengthAnalyzer(),
                        DiversityAnalyzer(),
                        ErrorCaseAnalyzer(),
                        ReadabilityAnalyzer(),
                    ],
                ),
            ).run()


def run_llm(dataset_loaders: list[DatasetLoader], run_dir: RunPaths):

    for i, dataset_loader in enumerate(dataset_loaders):
        _, _, test = dataset_loader.load_pairs()

        run_dir.pipeline_dir = f"LLM_Zeroshot_{i}"

        LLMEvaluationPipeline(
            model_config=ZeroShotLLMConfig(),
            generation_config=GenerationConfig(
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
        ).run(test)


def main():
    traingings_configs: list[TrainingConfig] = [
        training_config_1,
        training_config_2,
        TrainingConfig(),
    ]

    generation_configs: list[GenerationConfig] = [
        generation_config_1,
        GenerationConfig(),
        GenerationConfig(length_penalty=0.9, no_repeat_ngram_size=3, repetition_penalty=1.1),
    ]

    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=10000, max_eval_samples=2000),
        WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        OneStopLoader(),
    ]

    run_dir = RunPaths.for_runs_root()

    run_finetuning(traingings_configs, generation_configs, dataset_loaders, run_dir)

    run_llm(dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

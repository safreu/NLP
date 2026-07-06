from configuration.llm_config import (
    LLMTrainingConfig,
    llm_training_config_1,
    llm_training_config_2,
    llm_generation_config_1_sampling,
    llm_generation_config_1_beam,
    llm_generation_config_1_contrastive,
    llm_generation_config_1_greedy
)
from configuration.seq2seq_config import (
    GenerationConfig,
    TrainingConfig,
    training_config_1,
    training_config_2,
    generation_config_1
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
from pipeline.llm_evalution_pipeline import LLMEvaluationPipeline
from pipeline.llm_training_pipeline import LLMTrainingPipeline
from pipeline.seq2seq_evaluation_pipeline import Seq2SeqEvaluationPipeline
from pipeline.seq2seq_training_pipeline import Seq2SeqTrainingPipeline
from storage.paths import RunPaths
from pathlib import Path


def run_finetuning_seq2seq(
    trainings_configs: list[TrainingConfig],
    generation_configs: list[GenerationConfig],
    dataset_loaders: list[DatasetLoader],
    run_dir: RunPaths,
):
    for dataset_loader in dataset_loaders:
        dataset_name = dataset_loader.__class__.__name__.replace("Loader", "")

        for train_idx, train_conf in enumerate(trainings_configs):
            run_name = f"{dataset_name}_train{train_idx}"

            Seq2SeqTrainingPipeline(
                name=run_name,
                dataset_loader=dataset_loader,
                training_config=train_conf,
                run_paths=run_dir,
                evaluation_pipeline=Seq2SeqEvaluationPipeline(
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
                ),
            ).run()

def main():

    # training_configs: list[TrainingConfig] = [
    # TrainingConfig(
    # num_train_epochs=3,
    # learning_rate=5e-5,
    # weight_decay=0.01,
    # warmup_steps=500,
    # ),
    # TrainingConfig(
    # num_train_epochs=5,
    # learning_rate=5e-5,
    # weight_decay=0.01,
    # warmup_steps=500,
    # ),
    # TrainingConfig(
    # num_train_epochs=3,
    # learning_rate=1e-4,
    # weight_decay=0.01,
    # warmup_steps=500,
    # ),
    # TrainingConfig(
    # num_train_epochs=5,
    # learning_rate=1e-4,
    # weight_decay=0.01,
    # warmup_steps=500,
    # ),
    # TrainingConfig(
    # num_train_epochs=3,
    # learning_rate=2e-4,
    # weight_decay=0.01,
    # warmup_steps=500,
    # ),
    # TrainingConfig(
    # num_train_epochs=5,
    # learning_rate=2e-4,
    # weight_decay=0.01,
    # warmup_steps=500,
    # ),
    # ]

    # generation_configs: list[GenerationConfig] = [
    # GenerationConfig(
    # max_new_tokens=256,
    # do_sample=False,
    # num_beams=1,
    # ),
    # GenerationConfig(
    # max_new_tokens=256,
    # do_sample=False,
    # num_beams=4,
    # length_penalty=1.0,
    # no_repeat_ngram_size=3,
    # ),
    # GenerationConfig(
    # max_new_tokens=256,
    # do_sample=False,
    # num_beams=4,
    # length_penalty=0.9,
    # no_repeat_ngram_size=3,
    # repetition_penalty=1.1,
    # ),
    # GenerationConfig(
    # max_new_tokens=256,
    # do_sample=False,
    # num_beams=6,
    # length_penalty=0.9,
    # no_repeat_ngram_size=3,
    # repetition_penalty=1.1,
    # ),
    # ]

    EPOCHS = [5, 8, 10, 12, 15, 18, 20]
    LEARNING_RATES = [5e-5, 1e-4, 2e-4]

    training_configs: list[TrainingConfig] = [
        TrainingConfig(
            num_train_epochs=epochs,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=500,
        )
        for lr in LEARNING_RATES
        for epochs in EPOCHS
    ]

    generation_configs: list[GenerationConfig] = [
        GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        ),
        GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            num_beams=4,
            length_penalty=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        ),
        GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            num_beams=6,
            length_penalty=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        ),
    ]

    # trainings_configs: list[TrainingConfig] = [
    #    training_config_1,
    #    training_config_2,
    #    TrainingConfig(),
    # ]

    # generation_configs: list[GenerationConfig] = [
    #    generation_config_1,
    #    GenerationConfig(),
    #    GenerationConfig(length_penalty=0.9, no_repeat_ngram_size=3, repetition_penalty=1.1),
    # ]

    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=10000, max_eval_samples=2000),
        WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        OneStopLoader(),
    ]

    run_dir = RunPaths.for_runs_root(Path("runs/seq2seq"))

    run_finetuning_seq2seq(training_configs, generation_configs, dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

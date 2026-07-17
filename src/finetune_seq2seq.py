import time
from pathlib import Path

from dotenv import load_dotenv

from configuration.seq2seq_config import (
    GenerationConfig,
    TrainingConfig,
    generation_config_1,  # noqa: F401
    training_config_1,  # noqa: F401
    training_config_2,  # noqa: F401
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
from pipeline.seq2seq_evaluation_pipeline import Seq2SeqEvaluationPipeline
from pipeline.seq2seq_training_pipeline import Seq2SeqTrainingPipeline
from storage.paths import RunPaths

load_dotenv()


def run_finetuning_seq2seq(
    trainings_configs: list[TrainingConfig],
    generation_configs: list[GenerationConfig],
    dataset_loaders: list[DatasetLoader],
    run_dir: RunPaths,
):
    for dataset_loader in dataset_loaders:
        start = time.time()
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
                    extra_evaluators=[AssetSariEvaluator(split="validation", max_examples=0)],
                ),
            ).run()

            print(
                f"{dataset_name} with {train_idx} finished in ",
                f"{(time.time() - start) / 60:.1f} minutes",
            )


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
    
    
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        num_beams=4,
        length_penalty=0.9,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
    )

    training_config_20_epochs = TrainingConfig(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=10,
        dataloader_num_workers=8,
        save_total_limit=2,
        seed=42,
    )


    training_config_5_epochs = TrainingConfig(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=10,
        dataloader_num_workers=8,
        save_total_limit=2,
        seed=42,
    )

    training_configs: list[TrainingConfig] = [
        training_config_20_epochs,
        training_config_5_epochs,
    ]

    generation_configs: list[GenerationConfig] = [
        generation_config
    ]

    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(max_train_samples=10000, max_eval_samples=2000),
        WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        OneStopLoader(),
    ]

    run_dir = RunPaths.for_runs_root(Path("runs/seq2seq/finetune"))

    run_finetuning_seq2seq(training_configs, generation_configs, dataset_loaders, run_dir)


if __name__ == "__main__":
    main()

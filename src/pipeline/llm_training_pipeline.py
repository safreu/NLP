from dataclasses import replace

from configuration.llm_config import LLMTrainingConfig
from data.dataset_loader import DatasetLoader
from pipeline.llm_evalution_pipeline import LLMEvaluationPipeline
from preprocessing.dataset_builder import to_dataset
from storage.paths import RunPaths
from training.llm_trainer import train_model


class LLMTrainingPipeline:
    def __init__(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        training_config: LLMTrainingConfig,
        run_paths: RunPaths,
        evaluation_pipeline: LLMEvaluationPipeline,
    ):
        self.name = name
        self.dataset_loader = dataset_loader
        self.config = training_config
        self.run_paths = run_paths
        self.evaluation_pipeline = evaluation_pipeline

    def run(self) -> None:
        self.run_paths.pipeline_dir = self.name

        print(f"Running Pipeline {self.name}")

        train, valid, test = self.dataset_loader.load_pairs(add_prompt=False)

        train_model(
            train=to_dataset(train),
            valid=to_dataset(valid),
            path=self.run_paths.model_dir,
            config=self.config,
        )
        
        self.evaluation_pipeline.model_config = replace(
            self.config,
            model_name=str(self.run_paths.model_dir)
        )

        self.evaluation_pipeline.run(test)

        print(f"Finished Pipeline {self.name}")

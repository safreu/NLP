from configuration.seq2seq_config import TrainingConfig
from data.dataset_loader import DatasetLoader
from pipeline.seq2seq_evaluation_pipeline import Seq2SeqEvaluationPipeline
from preprocessing.dataset_builder import to_dataset
from storage.paths import RunPaths
from training.trainer import train_model


class Seq2SeqTrainingPipeline:
    def __init__(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        training_config: TrainingConfig,
        run_paths: RunPaths,
        evaluation_pipeline: Seq2SeqEvaluationPipeline,
    ):
        self.name = name
        self.dataset_loader = dataset_loader
        self.config = training_config
        self.run_paths = run_paths
        self.evaluation_pipeline = evaluation_pipeline

    def run(self) -> None:
        self.run_paths.pipeline_dir = self.name

        print(f"Running Pipeline {self.name}")

        train, valid, test = self.dataset_loader.load_pairs()

        train_model(
            train=to_dataset(train),
            valid=to_dataset(valid),
            path=self.run_paths.model_dir,
            config=self.config,
        )

        self.evaluation_pipeline.run(test)

        self.config.save(self.run_paths.pipeline_dir)

        print(f"Finished Pipeline {self.name}")

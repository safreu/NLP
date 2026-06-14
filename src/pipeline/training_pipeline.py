from config import TrainingConfig
from data.dataset_loader import DatasetLoader
from pipeline.evaluation_pipeline import EvaluationPipeline
from preprocessing.dataset_builder import to_dataset
from storage.json_store import write_json
from storage.paths import RunPaths
from training.trainer import train_model


class TrainingPipeline:
    def __init__(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        config: TrainingConfig,
        run_paths: RunPaths,
        evaluation_pipeline: EvaluationPipeline
    ):
        self.name = name
        self.dataset_loader = dataset_loader
        self.config = config
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

        print(f"Finished Pipeline {self.name}")

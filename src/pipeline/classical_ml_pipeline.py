from pathlib import Path

from config import ClassicalMLConfig
from data.dataset_loader import DatasetLoader, Pair
from evaluation.classical_evaluate import evaluate_classical_model
from preprocessing.classical_training_data import to_classical_pairs
from storage.json_store import write_json
from training.classical_trainer import train_classical_model


class ClassicalMLPipeline:
    """Train and evaluate the local classical simplifier in the NLP2 run format."""

    def __init__(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        config: ClassicalMLConfig | None = None,
    ) -> None:
        self.name = name
        self.dataset_loader = dataset_loader
        self.config = config or ClassicalMLConfig()

    def run(self, run_dir: Path) -> None:
        pipeline_dir = run_dir / self.name
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        model_dir = pipeline_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running Pipeline {self.name}")
        train_pairs, valid_pairs, test_pairs = self.dataset_loader.load_pairs()
        train_pairs = self._limit(train_pairs, self.config.max_train_samples)
        valid_pairs = self._limit(valid_pairs, self.config.max_eval_samples)
        test_pairs = self._limit(test_pairs, self.config.max_eval_samples)
        train = to_classical_pairs(train_pairs)
        valid = to_classical_pairs(valid_pairs)
        test = to_classical_pairs(test_pairs)
        artifacts = train_classical_model(
            train_pairs=train,
            valid_pairs=valid,
            path=model_dir,
            config=self.config,
        )

        scores = {
            "validation": artifacts.validation_metrics,
            "test": evaluate_classical_model(
                test_pairs=test,
                artifacts=artifacts,
                predictions_path=pipeline_dir / "predictions.json",
                config=self.config,
            ),
        }
        write_json(scores, pipeline_dir / "scores.json")
        print(f"Finished Pipeline {self.name}")

    @staticmethod
    def _limit(pairs: list[Pair], limit: int | None) -> list[Pair]:
        return pairs if limit is None else pairs[:limit]

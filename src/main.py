
from config import TrainingConfig
from data.onestop_loader import OneStopLoader
from data.wikilarge_loader import WikiLargeLoader
from pipeline.training_pipeline import EvaluationMode, TrainingPipeline
from storage.run_store import create_run_dir

def main() -> None:
    experiment_dir = create_run_dir()
    
    TrainingPipeline(
        name="onestop",
        dataset_loader=OneStopLoader(),
        config=TrainingConfig(),
        evaluation_mode=EvaluationMode.CHECKPOINTS,
    ).run(experiment_dir)
    
    TrainingPipeline(
        name="wikilarge",
        dataset_loader=WikiLargeLoader(max_train_samples=10000, max_eval_samples=2000),
        config=TrainingConfig(epochs=3, max_target_length=128),
        evaluation_mode=EvaluationMode.CHECKPOINTS,
    ).run(experiment_dir)

if __name__ == "__main__":
    main()

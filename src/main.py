
from data.onestop_loader import OneStopLoader
from data.wikilarge_loader import WikiLargeLoader
from pipeline.training_pipeline import EvaluationMode, TrainingPipeline
from storage.run_store import create_run_dir

def main() -> None:
    experiment_dir = create_run_dir()
    
    TrainingPipeline(
        name="onestop",
        dataset_loader=OneStopLoader(),
        evaluation_mode=EvaluationMode.FINAL_MODEL,
    ).run(experiment_dir)
    
    TrainingPipeline(
        name="wikilarge",
        dataset_loader=WikiLargeLoader(),
        evaluation_mode=EvaluationMode.FINAL_MODEL,
    ).run(experiment_dir)

if __name__ == "__main__":
    main()

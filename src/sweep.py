from config import (
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
from pipeline.evaluation_pipeline import EvaluationPipeline
from pipeline.training_pipeline import TrainingPipeline
from storage.paths import RunPaths


def main():
    traingings_configs: list[TrainingConfig] = [
        training_config_1, 
        training_config_2, 
        TrainingConfig(),
    ]
    
    generation_configs: list[GenerationConfig] = [
        generation_config_1,
        GenerationConfig(),
        GenerationConfig(),
    ]
    
    dataset_loaders: list[DatasetLoader] = [
        NewselaLoader(),
        WikiLargeLoader(),
        OneStopLoader(),       
    ]
   
    run_dir = RunPaths.for_runs_root()
    
    for i in range(3):
        for j in range(3):
            TrainingPipeline(
                name=f"Config_{i}",
                dataset_loader=dataset_loaders[j],
                training_config=traingings_configs[i],
                run_paths=run_dir,
                evaluation_pipeline=EvaluationPipeline(
                    generation_config=generation_configs[i],
                    run_paths=run_dir,
                    analyzers=[
                        CopyAnalyzer(threshold=0.95),
                        InformationLossAnalyzer(),
                        LengthAnalyzer(),
                        DiversityAnalyzer(),
                        ErrorCaseAnalyzer(),
                        ReadabilityAnalyzer(),
                    ]
                )
            ).run()


if __name__ == "__main__":
    main()
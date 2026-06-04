from pathlib import Path
from enum import Enum

from data.dataset_loader import DatasetLoader
from evaluation.checkpoint_compare import compare_best_checkpoints
from evaluation.prediction_analysis import analyze_prediction_copies
from preprocessing.dataset_builder import to_dataset
from training.trainer import train_model
from evaluation.evaluate import evaluate_checkpoints, evaluate_model
from storage.json_store import write_json

class EvaluationMode(Enum):
    FINAL_MODEL = "final_model"
    CHECKPOINTS = "checkpoints"

class TrainingPipeline:
    def __init__(self, name: str, dataset_loader: DatasetLoader, evaluation_mode: EvaluationMode=EvaluationMode.FINAL_MODEL):
        self.name = name
        self.dataset_loader = dataset_loader
        self.evaluation_mode = evaluation_mode
        
        
    def _evaluate(self, test, model_dir, pipeline_dir):
        predictions_path = pipeline_dir / "predictions.json"
        scores_path = pipeline_dir / "scores.json"
        
        if self.evaluation_mode == EvaluationMode.CHECKPOINTS:
            results = evaluate_checkpoints(test, model_dir)
            
            write_json(results, scores_path)
            
            compare_best_checkpoints(
                scores_path=scores_path,
                model_dir=model_dir,
                output_path=pipeline_dir/"best_checkpoints_comparison.json",
                metric_path="sari",
                k=5,
                higher_is_better=True,
            )
            return
        
        results = evaluate_model(
                test_pairs=test,
                model_path=model_dir,
                predictions_path=predictions_path,
            )
            
        write_json(results, scores_path)
        
        analyze_prediction_copies(
            predictions_path=predictions_path,
            output_path=pipeline_dir / "copy_analysis.json",
            near_copy_threshold=0.95,
        )
        
    
    def run(self, run_dir: Path) -> None:
        pipeline_dir = run_dir / self.name
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = pipeline_dir / "model"

        
        print(f"Running Pipeline {self.name}")
        
        train, valid, test = self.dataset_loader.load_pairs()
        
        train_dataset = to_dataset(train)
        valid_dataset = to_dataset(valid)
        
        train_model(
            train=train_dataset,
            valid=valid_dataset,
            path=model_dir,
        )
        
        self._evaluate(test, model_dir, pipeline_dir)
    
        
        print(f"Finished Pipeline {self.name}")
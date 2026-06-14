from abc import ABC, abstractmethod
from pathlib import Path
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow

class PredictionAnalyzer(ABC):
    
    @abstractmethod
    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        pass
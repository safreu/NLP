from pathlib import Path
from typing import Protocol

from transformers import GenerationConfig


class MetricsEvaluator(Protocol):
    def compute(
        self, 
        model_path: Path,
        generation_config: GenerationConfig, 
    ) -> dict:
        ...
        
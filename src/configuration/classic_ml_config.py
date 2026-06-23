from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassicalMLConfig:
    model_type: str = "logistic_regression"
    random_state: int = 42
    lowercase: bool = True
    min_replacement_count: int = 1
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    classifier_parameters: dict[str, Any] | None = None
    compute_generation_metrics: bool = True

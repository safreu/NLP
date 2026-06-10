from dataclasses import dataclass
from typing import Any

MIN_LENGTH_RATIO = 0.2
SIMILARITY_THRESHOLD = 0.9
SEED = 42


@dataclass
class TrainingConfig:
    model_name: str = "google/flan-t5-base"
    max_input_length: int = 256
    max_target_length: int = 256
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 2e-4
    save_total_limit: int = 2
    seed: int = 42

    num_beams: int = 4
    length_penalty: float = 0.9
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.1

    @property
    def generation_config(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_target_length,
            "do_sample": False,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
        }


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

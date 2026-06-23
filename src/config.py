from dataclasses import asdict, dataclass
from typing import Any

MIN_LENGTH_RATIO = 0.2
SIMILARITY_THRESHOLD = 0.9
SEED = 42


@dataclass
class TrainingConfig:
    model_name: str = "google/flan-t5-base"

    max_input_length: int = 256
    max_target_length: int = 256

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 15
    learning_rate: float = 2e-4

    weight_decay: float | None = None
    adam_epsilon: float | None = None
    warmup_steps: int | None = None

    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    predict_with_generate: bool = True
    logging_steps: int = 10
    dataloader_num_workers: int = 8

    save_total_limit: int = 2
    seed: int = SEED

    def to_dict(self) -> dict[str, Any]:
        ignored = {"model_name", "max_input_length", "max_target_length"}
        return {k: v for k, v in asdict(self).items() if v is not None and k not in ignored}


@dataclass
class GenerationConfig:
    max_new_tokens: int | None = None
    do_sample: bool = False
    num_beams: int | None = None  # 8

    length_penalty: float | None = None  # 0.9
    no_repeat_ngram_size: int | None = None  # 3
    repetition_penalty: float | None = None  # 1.1

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


training_config_1 = TrainingConfig(
    # Page 343, Section 4.3 Training Details:
    # https://aclanthology.org/2021.inlg-1.38.pdf
    model_name="t5-base",
    max_input_length=256,
    max_target_length=256,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_epsilon=1e-8,
    warmup_steps=5,
    num_train_epochs=5,
    seed=12,
)

generation_config_1 = GenerationConfig(
    # Page 343, Section 4.3 Training Details:
    # https://aclanthology.org/2021.inlg-1.38.pdf
    num_beams=8,
)

training_config_2 = TrainingConfig(
    # Page 4, Section 4 The Training Procedure
    # https://www.researchgate.net/profile/Ramazan_Mengi/publication/
    # 376232167_Fine-tuning_T5_and_RoBERTa_Models_for_Enhanced_Text_
    # Summarization_and_Sentiment_Analysis/
    # links/656f7463fd4c91437ba4df31/
    # Fine-tuning-T5-and-RoBERTa-Models-for-Enhanced-Text-Summarization-and-Sentiment-Analysis.pdf
    model_name="t5-base",
    num_train_epochs=5,
    max_target_length=448,
    learning_rate=2e-5,
)


@dataclass
class ZeroShotLLMConfig:
    model_name: str = "google/gemma-4-12b-it"
    revision: str | None = None
    device: str | None = None


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

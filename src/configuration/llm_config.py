from dataclasses import asdict, dataclass


@dataclass
class LLMTrainingConfig:
    model_name: str = "google/gemma-4-12b-it"

    max_seq_length: int | None = None
    num_proc: int | None = None

    use_qlora: bool | None = None
    packing: bool | None = None

    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    per_device_train_batch_size: int | None = None
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    num_train_epochs: int | None = None

    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    bf16: bool = True
    report_to: str = "none"

    def to_dict(self):
        ignored = {
            "model_name",
            "max_seq_length",
            "num_proc",
            "use_qlora",
            "packing",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        }

        return {k: v for k, v in asdict(self).items() if v is not None and k not in ignored}


@dataclass
class ZeroShotLLMConfig:
    model_name: str = "google/gemma-4-12b-it"
    revision: str | None = None
    device: str | None = None

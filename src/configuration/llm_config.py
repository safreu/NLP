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

    warmup_steps: int | None = None
    weight_decay: float | None = None
    save_steps: int | None = None
    optim: str | None = None
    gradient_checkpointing: bool | None = None

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
class LLMGenerationConfig:
    max_new_tokens: int | None = None
    do_sample: bool | None = None
    num_beams: int | None = None
    no_repeat_ngram_size: int | None = None

    early_stopping: bool | None = None

    top_p: float | None = None
    top_k: int | None = None
    temperature: float | None = None

    penalty_alpha: float | None = None

    repetition_penalty: float | None = None

    def to_dict(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


llm_training_config_1 = LLMTrainingConfig(
    model_name="google/gemma-4-E4B-it",
    use_qlora=True,
    learning_rate=2e-5,
    weight_decay=0.05,
    per_device_train_batch_size=2,
    num_train_epochs=3,
)

llm_generation_config_1_greedy = LLMGenerationConfig(
    no_repeat_ngram_size=5,
    max_new_tokens=1024,
    do_sample=False,
)

llm_generation_config_1_beam = LLMGenerationConfig(
    no_repeat_ngram_size=5,
    max_new_tokens=1024,
    num_beams=5,
    early_stopping=True,
)

llm_generation_config_1_sampling = LLMGenerationConfig(
    no_repeat_ngram_size=5,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    top_k=5,
    temperature=0.5,
)

llm_generation_config_1_contrastive = LLMGenerationConfig(
    no_repeat_ngram_size=5,
    max_new_tokens=1024,
    penalty_alpha=0.05,
    top_k=5,
)

llm_training_config_2 = LLMTrainingConfig(
    model_name="google/gemma-4-E2B-it",
    
    use_qlora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    
    max_seq_length=512,
    
    warmup_steps=500,
    bf16=True,
    save_steps=500,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit"
)

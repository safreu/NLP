from pathlib import Path

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from configuration.llm_config import LLMTrainingConfig


def load_model(config: LLMTrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None

    if config.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    if config.use_qlora:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def format_template(text, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"Simplify this text:\n\n{text['input']}",
        },
        {
            "role": "assistant",
            "content": text["target"],
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def format_dataset(dataset, tokenizer, config: LLMTrainingConfig):
    def map_text(text):
        return {
            "text": format_template(text, tokenizer),
        }

    return dataset.map(map_text, remove_columns=dataset.column_names, num_proc=config.num_proc)


def create_lora_config(config: LLMTrainingConfig):
    if not config.use_qlora:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def create_trainings_args(path: Path | str, config: LLMTrainingConfig):
    return SFTConfig(
        output_dir=str(path),
        max_length=config.max_seq_length,
        packing=config.packing,
        **config.to_dict(),
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    path: Path | str,
    config: LLMTrainingConfig,
):
    return SFTTrainer(
        model=model,
        args=create_trainings_args(path, config),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        peft_config=create_lora_config(config),
    )


def save_model(trainer, tokenizer, path: Path | str):
    trainer.model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_model(train, valid, path: Path | str, config: LLMTrainingConfig):
    model, tokenizer = load_model(config)

    train_format = format_dataset(train, tokenizer, config)
    valid_format = format_dataset(valid, tokenizer, config)

    trainer = create_trainer(model, tokenizer, train_format, valid_format, path, config)

    trainer.train()

    save_model(trainer, tokenizer, path)

    return trainer

from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from configuration.seq2seq_config import TrainingConfig

torch.set_num_threads(16)


def load_model(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    return model, tokenizer


def tokenize_batch(batch, tokenizer, config: TrainingConfig):
    model_inputs = tokenizer(batch["input"], max_length=config.max_input_length, truncation=True)

    labels = tokenizer(
        text_target=batch["target"], max_length=config.max_target_length, truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def tokenize_dataset(dataset, tokenizer, config: TrainingConfig):
    return dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, config), batched=True, num_proc=16
    )


def create_training_args(path: Path | str, config: TrainingConfig):
    return Seq2SeqTrainingArguments(
        output_dir=str(path),
        **config.to_dict(),
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    path: Path | str,
    config: TrainingConfig,
):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    return Seq2SeqTrainer(
        model=model,
        args=create_training_args(path, config),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )


def save_model(model, tokenizer, path: Path | str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_model(train, valid, path: Path | str, config: TrainingConfig):
    model, tokenizer = load_model(config)

    train_tokenized = tokenize_dataset(train, tokenizer, config)
    valid_tokenized = tokenize_dataset(valid, tokenizer, config)

    trainer = create_trainer(model, tokenizer, train_tokenized, valid_tokenized, path, config)

    trainer.train()

    save_model(model, tokenizer, path)

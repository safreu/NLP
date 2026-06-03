from config import (
    MODEL_NAME, 
    MODEL_OUTPUT_DIR, 
    MAX_INPUT_LENGTH, 
    MAX_TARGET_LENGTH,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

import torch
torch.set_num_threads(16)

def load_model(model_name: str=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return model, tokenizer


def tokenize_batch(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        num_proc=16
    )


def create_training_args(path: str):
    return Seq2SeqTrainingArguments(
        output_dir=path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        logging_steps=10,
        save_total_limit=None,
        dataloader_num_workers=8
    )
    
    
def create_trainer(model, tokenizer, train_dataset, valid_dataset, path: str):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    
    return Seq2SeqTrainer(
        model=model,
        args=create_training_args(path),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    
def save_model(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    

def train_model(train, valid, path: str=MODEL_OUTPUT_DIR):
    model, tokenizer = load_model()
    
    train_tokenized = tokenize_dataset(train, tokenizer)
    valid_tokenized = tokenize_dataset(valid, tokenizer)
    
    trainer = create_trainer(model, tokenizer, train_tokenized, valid_tokenized, path)
    
    trainer.train()
    
    save_model(model, tokenizer, path)
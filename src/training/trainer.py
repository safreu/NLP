from datasets import DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL_NAME = "google/flan-t5-small"

def train_model(train, valid):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=256,
            truncation=True
        )

        labels = tokenizer(
            text_target=batch["target"],
            max_length=256,
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    train_token = train.map(preprocess, batched=True)
    valid_token = valid.map(preprocess, batched=True)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
    
    args = Seq2SeqTrainingArguments(
        output_dir="models/text-simplifier/OneStop",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        predict_with_generate=True,
        logging_steps=10,
        save_total_limit=2
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_token,
        eval_dataset=valid_token,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    
    trainer.save_model("models/text-simplifier/OneStop")
    tokenizer.save_pretrained("models/text-simplifier/OneStop")
    
    

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "t5-small"
DATASET_NAME = "an-atlas/wikilarge"
TASK_PREFIX = "simplify: " 

OUTPUT_DIR = "./t5-small-wikilarge"  

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 3e-4

MAX_TRAIN_SAMPLES: int | None = None
MAX_EVAL_SAMPLES: int | None = 2000

SEED = 42
                  #
def _load_bertscore():
    try:
        here = Path(__file__).resolve()
    except NameError:  # __file__ is undefined when pasted into a notebook cell
        here = Path.cwd().resolve()

    for parent in [here, *here.parents]:
        for candidate in (
            parent / "src" / "metrics" / "bertScore.py",
            parent / "metrics" / "bertScore.py",
        ):
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("bertScore", candidate)
                module = importlib.util.module_from_spec(spec)
                sys.modules["bertScore"] = module
                spec.loader.exec_module(module)
                return module.compute_bertscore

    raise FileNotFoundError(f"Could not locate src/metrics/bertScore.py from {here}")


compute_bertscore = _load_bertscore()

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
USE_FP16 = DEVICE == "cuda"
print(f"Using device: {DEVICE} (fp16={USE_FP16})")


# load dataset                                                         #
print(f"\nLoading dataset '{DATASET_NAME}' ...")
dataset = load_dataset(DATASET_NAME)
print(dataset)

if MAX_TRAIN_SAMPLES is not None:
    dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))
if MAX_EVAL_SAMPLES is not None:
    n_test = min(MAX_EVAL_SAMPLES, len(dataset["test"]))
    dataset["test"] = dataset["test"].select(range(n_test))

# load model + tokenizer                                                   #
print(f"\nLoading model '{MODEL_NAME}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# simple preprocessing with token normalization                
def preprocess(batch: dict[str, list[str]]) -> dict:
    inputs = [TASK_PREFIX + text for text in batch["Normal"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["Simple"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("\nTokenizing splits ...")
tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# fine-tune                                                                #
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    fp16=USE_FP16,
    predict_with_generate=True,
    logging_steps=100,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("\nStarting training ...")
trainer.train()

print(f"\nSaving fine-tuned model to '{OUTPUT_DIR}/final' ...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# generate for the test split
def simplify_sentences(sentences: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    """Run the fine-tuned model over raw sentences and return simplifications."""
    model.eval()
    model.to(DEVICE)
    predictions: list[str] = []

    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        enc = tokenizer(
            [TASK_PREFIX + s for s in chunk],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return predictions


print("\nGenerating simplifications for the test split ...")
test_normal: list[str] = dataset["test"]["Normal"]
test_simple: list[str] = dataset["test"]["Simple"] 

test_predictions = simplify_sentences(test_normal)


# evaluate
print("\nComputing BERTScore (model output vs. reference simplification) ...")
result = compute_bertscore(
    standard=test_predictions,  # candidates
    simple=test_simple,         # references
    lang="en",
    device=DEVICE,
    verbose=True,
)

print("\n" + "=" * 60)
print("BERTScore on the WikiLarge test split")
print("=" * 60)
print(f"  Precision (mean): {result['precision_mean']:.4f}")
print(f"  Recall    (mean): {result['recall_mean']:.4f}")
print(f"  F1        (mean): {result['f1_mean']:.4f}")
print("=" * 60)

# few examples
print("\nExamples:")
for i in range(min(5, len(test_predictions))):
    print(f"\n[{i}]")
    print(f"  Normal    : {test_normal[i]}")
    print(f"  Reference : {test_simple[i]}")
    print(f"  Predicted : {test_predictions[i]}")
    print(f"  BERTScore F1: {result['f1'][i]:.4f}")

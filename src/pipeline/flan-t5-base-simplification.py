"""Historical end-to-end FLAN-T5 simplification script.

This is the original monolithic experiment used to fine-tune
``google/flan-t5-base`` on the text simplification corpora and evaluate it on
the test split plus ASSET. It predates the structured experiment runner and is
kept only as a reference for how the first results were produced.

Do not build new experiments on top of this file. The supported entry points
are:

* ``uv run python -m main``
  The experiment CLI (``src/main.py``). It wires dataset loaders,
  training pipeline and config/run-directory handling together. 
  Use this to train and evaluate.
* ``uv run python -m pipeline.sari_asset_pipeline`` 
  standalone ASSET SARI evaluation of an already-trained model (see ``pipeline/COMMANDS.md``).

"""

from __future__ import annotations

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


def _find_src() -> Path:
    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd().resolve()

    for parent in [here, *here.parents]:
        if (parent / "preprocessing").is_dir() and (parent / "metrics").is_dir():
            return parent

    raise FileNotFoundError("Could not locate the 'src' package root")


SRC_DIR = _find_src()
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics_builder import compute_all_metrics
from preprocessing.cleaner import clean_text
from preprocessing.dataset_builder import split_pairs, to_dataset
from preprocessing.filter import length_ratio, text_similarity
from storage.json_store import write_json
from storage.prediction_store import prediction_rows
from storage.run_store import create_run_dir

MODEL_NAME = "google/flan-t5-base"
TASK_PREFIX = "simplify: "

OUTPUT_DIR = "./flan-t5-base-simplification"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# preprocessing filter thresholds (mirrors data.one_stop_english_corpus)
MAX_SIMILARITY = 0.8
MIN_LENGTH_RATIO = 0.2

# optional caps to keep runtime manageable (None = use everything)
MAX_TRAIN_SAMPLES: int | None = None
MAX_EVAL_SAMPLES: int | None = None

SEED = 42


# dataset loaders                                                          #
# each returns a list of raw (complex, simple) pairs
def load_onestop() -> list[tuple[str, str]]:
    from data.one_stop_english_corpus import OneStopEnglish

    path = str(REPO_ROOT / "data" / "OneStopEnglishCorpus" / "Sentence-Aligned")
    corpus = OneStopEnglish.load_from_disk(path)
    return [(entry.source, entry.target) for entry in corpus.entries]


def load_wikilarge() -> list[tuple[str, str]]:
    dataset = load_dataset("an-atlas/wikilarge")
    return [(row["Normal"], row["Simple"]) for row in dataset["train"]]


def load_wikismall() -> list[tuple[str, str]]:
    dataset = load_dataset("cestwc/adapted-wikismall")
    return [(row["long"], row["short"]) for row in dataset["train"]]


DATASET_LOADERS = {
    "OneStopEnglish": load_onestop,
    "WikiLarge": load_wikilarge,
    "WikiSmall": load_wikismall,
}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# apply preprocessing from the preprocessing folder
def preprocess_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    processed: list[tuple[str, str]] = []

    for source, target in pairs:
        clean_source = clean_text(source)
        clean_target = clean_text(target)

        if not clean_source or not clean_target:
            continue
        if text_similarity(clean_source, clean_target) > MAX_SIMILARITY:
            continue
        if length_ratio(clean_source, clean_target) < MIN_LENGTH_RATIO:
            continue

        pair = (TASK_PREFIX + clean_source, clean_target)
        if pair in seen:
            continue

        seen.add(pair)
        processed.append(pair)

    return processed


def simplify(
    inputs: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    # inputs already include the task prefix
    model.eval()
    model.to(device)
    predictions: list[str] = []

    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        enc = tokenizer(
            chunk,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return predictions


# evaluation-only benchmark: multi-reference SARI on ASSET
def evaluate_on_asset(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> float:
    from metrics.metric_sari import sari

    dataset = load_dataset("facebook/asset", "simplification")["test"]
    asset_sources = [clean_text(row["original"]) for row in dataset]
    asset_references = [
        [clean_text(simplification) for simplification in row["simplifications"]] for row in dataset
    ]
    asset_predictions = simplify(
        [TASK_PREFIX + source for source in asset_sources],
        model,
        tokenizer,
        device,
    )

    score = float(
        sari.compute(
            sources=asset_sources,
            predictions=asset_predictions,
            references=asset_references,
        )["sari"]
    )
    print("\n" + "=" * 60)
    print(f"ASSET multi-reference SARI (test, {len(asset_sources)} sentences): {score:.4f}")
    print("=" * 60)
    return score


def main() -> None:
    device = select_device()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = False
    print(f"Using device: {device} (bf16={use_bf16})")

    # load every dataset, skipping any loader that fails
    print("\nLoading datasets from the data folder ...")
    raw_pairs: list[tuple[str, str]] = []
    for name, loader in DATASET_LOADERS.items():
        try:
            loaded = loader()
            raw_pairs.extend(loaded)
            print(f"  [ok]   {name}: {len(loaded)} pairs")
        except Exception as error:
            print(f"  [skip] {name}: {type(error).__name__}: {error}")

    if not raw_pairs:
        raise RuntimeError("No dataset could be loaded.")
    print(f"Total raw pairs: {len(raw_pairs)}")

    print("\nApplying preprocessing ...")
    pairs = preprocess_pairs(raw_pairs)
    print(f"Pairs after preprocessing: {len(pairs)}")

    # show one before/after example so the transformation is visible
    raw_example = raw_pairs[0]
    print("\nPreprocessing example:")
    print("  before:")
    print(f"    source: {raw_example[0]!r}")
    print(f"    target: {raw_example[1]!r}")
    print("  after:")
    print(f"    input : {(TASK_PREFIX + clean_text(raw_example[0]))!r}")
    print(f"    target: {clean_text(raw_example[1])!r}")

    # split + build datasets
    train_pairs, valid_pairs, test_pairs = split_pairs(pairs, random_state=SEED)

    if MAX_TRAIN_SAMPLES is not None:
        train_pairs = train_pairs[:MAX_TRAIN_SAMPLES]
    if MAX_EVAL_SAMPLES is not None:
        test_pairs = test_pairs[:MAX_EVAL_SAMPLES]

    train_dataset = to_dataset(train_pairs)
    valid_dataset = to_dataset(valid_pairs)
    print(
        f"\nSplits -> train: {len(train_pairs)}, valid: {len(valid_pairs)}, test: {len(test_pairs)}"
    )

    # load model + tokenizer
    print(f"\nLoading model '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict:
        model_inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing splits ...")
    tokenized_train = train_dataset.map(
        tokenize_batch, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_valid = valid_dataset.map(
        tokenize_batch, batched=True, remove_columns=valid_dataset.column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # fine-tune
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=use_bf16,
        predict_with_generate=False,
        logging_steps=100,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("\nStarting training ...")
    trainer.train()

    print(f"\nSaving fine-tuned model to '{OUTPUT_DIR}/final' ...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    # generate for the test split
    print("\nGenerating simplifications for the test split ...")
    test_inputs = [pair[0] for pair in test_pairs]
    test_targets = [pair[1] for pair in test_pairs]
    sources = [inp.removeprefix(TASK_PREFIX) for inp in test_inputs]

    predictions = simplify(test_inputs, model, tokenizer, device)

    # evaluate with all metrics from the metrics folder
    print("\nComputing all metrics ...")
    results = compute_all_metrics(sources, predictions, test_targets)

    print("\n" + "=" * 60)
    print("Evaluation on the test split")
    print("=" * 60)
    print(f"  BERTScore F1 (mean): {results['bert']['f1_mean']:.4f}")
    print(f"  BLEU               : {results['bleu']:.4f}")
    print(f"  Token F1 (mean)    : {results['f1']['f1_mean']:.4f}")
    print(f"  Flesch-Kincaid     : {results['flesch']['mean']:.4f}")
    print(f"  SARI               : {results['sari']:.4f}")
    print(f"  ROUGE-L            : {results['rouge-l']:.4f}")
    print("=" * 60)

    # few examples
    print("\nExamples:")
    for i in range(min(5, len(predictions))):
        print(f"\n[{i}]")
        print(f"  Source    : {sources[i]}")
        print(f"  Reference : {test_targets[i]}")
        print(f"  Predicted : {predictions[i]}")

    print("\nEvaluating on ASSET ...")
    try:
        asset_sari = evaluate_on_asset(model, tokenizer, device)
    except Exception as error:
        print(f"[skip] ASSET evaluation: {type(error).__name__}: {error}")
        asset_sari = None

    # save results as JSON                                                    #
    run_dir = create_run_dir()

    scores = {
        "model": OUTPUT_DIR,
        "num_test_sentences": len(predictions),
        "test_split": {
            "bertscore_precision_mean": results["bert"]["precision_mean"],
            "bertscore_recall_mean": results["bert"]["recall_mean"],
            "bertscore_f1_mean": results["bert"]["f1_mean"],
            "bleu": results["bleu"],
            "token_f1_mean": results["f1"]["f1_mean"],
            "flesch_kincaid_mean": results["flesch"]["mean"],
            "sari": results["sari"],
            "rouge_l": results["rouge-l"],
        },
        "asset_multi_reference_sari": asset_sari,
    }

    write_json(scores, run_dir / "scores.json")
    write_json(prediction_rows(sources, predictions, test_targets), run_dir / "predictions.json")
    print(f"\nAll results saved under: {run_dir}")


if __name__ == "__main__":
    main()

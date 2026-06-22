# mypy: ignore-errors
"""Final report-oriented evaluation for neural replacement filter outputs."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = find_project_root()
FILTER_ROOT = PROJECT_ROOT / "results" / "neural_replacement_filter"
FUNCTION_WORD_DIR = FILTER_ROOT / "outputs" / "function_word_safe"
CONTENT_QUALITY_DIR = FILTER_ROOT / "outputs" / "content_quality_safe"
DEFAULT_OUTPUT_DIR = FILTER_ROOT / "outputs" / "final_filter_evaluation"
DEFAULT_LOG_DIR = FILTER_ROOT / "logs" / "final_filter_evaluation"

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NUMBER_RE = re.compile(r"(?<!\w)[-+]?(?:\d{1,3}(?:,\s*\d{3})+|\d+)(?:\s*\.\s*\d+)?\s*%?(?!\w)")

OUTPUT_COLUMN_CANDIDATES = [
    "original_logistic_regression_output",
    "model_output_sentence",
    "logistic_regression_output",
    "original_output",
]

SUMMARY_COLUMNS = [
    "system",
    "selected_threshold",
    "num_sentences",
    "avg_sentence_length",
    "avg_token_count",
    "avg_word_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "avg_replacements_per_sentence",
    "percent_sentences_changed",
    "source_token_overlap",
    "reference_token_overlap",
    "bleu",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "sari",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "entity_preservation_rate",
    "number_preservation_rate",
]


class EvaluationLog:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)
        logging.info(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        self.lines.append(f"WARNING: {message}")
        logging.warning(message)

    def write(self, path: Path) -> None:
        sections = ["Final Filter Evaluation Log", "=" * 29, "", *self.lines]
        if self.warnings:
            sections.extend(["", "Warnings", "-" * 8, *self.warnings])
        path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def tokens(text: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def words(text: Any) -> list[str]:
    return WORD_RE.findall(str(text))


def token_f1(left: Any, right: Any) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def count_syllables_in_word(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    groups = re.findall(r"[aeiouy]+", cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_scores(texts: list[str]) -> tuple[float, float]:
    total_sentences = 0
    total_words = 0
    total_syllables = 0
    for text in texts:
        word_list = words(text)
        sentence_count = len(re.findall(r"[.!?]+", str(text))) or (1 if word_list else 0)
        total_sentences += sentence_count
        total_words += len(word_list)
        total_syllables += sum(count_syllables_in_word(word) for word in word_list)
    if total_sentences == 0 or total_words == 0:
        return 0.0, 0.0
    words_per_sentence = total_words / total_sentences
    syllables_per_word = total_syllables / total_words
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return float(flesch), float(grade)


def normalize_text(text: Any) -> str:
    return " ".join(tokens(text))


def normalize_number(number: str) -> str:
    number = number.strip()
    has_percent = number.replace(" ", "").endswith("%")
    core = number.rstrip("%") if has_percent else number
    core = re.sub(r"\s+", "", core.replace(",", ""))
    if "." in core:
        with suppress(ValueError):
            core = str(float(core)).rstrip("0").rstrip(".")
    return f"{core}%" if has_percent else core


def extract_numbers(text: Any) -> list[str]:
    return [normalize_number(match.group(0)) for match in NUMBER_RE.finditer(str(text))]


def preservation_rate(source_items: list[str], output_items: list[str]) -> float | None:
    if not source_items:
        return None
    source_counts = Counter(source_items)
    output_counts = Counter(output_items)
    preserved = sum(min(count, output_counts[item]) for item, count in source_counts.items())
    return preserved / sum(source_counts.values())


def number_preservation_rate(sources: list[str], outputs: list[str]) -> float | None:
    rates = [
        rate
        for source, output in zip(sources, outputs, strict=True)
        if (rate := preservation_rate(extract_numbers(source), extract_numbers(output))) is not None
    ]
    return float(sum(rates) / len(rates)) if rates else None


def load_spacy_for_entities(log: EvaluationLog):
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception as exc:
        log.warning(f"Entity preservation unavailable because spaCy model loading failed: {exc}")
        return None


def entity_preservation_rate(
    sources: list[str], outputs: list[str], log: EvaluationLog
) -> float | None:
    nlp = load_spacy_for_entities(log)
    if nlp is None:
        return None
    rates: list[float] = []
    for source, output in zip(sources, outputs, strict=True):
        source_entities = [ent.text.lower() for ent in nlp(source).ents]
        output_entities = [ent.text.lower() for ent in nlp(output).ents]
        rate = preservation_rate(source_entities, output_entities)
        if rate is not None:
            rates.append(rate)
    return float(sum(rates) / len(rates)) if rates else None


def find_output_column(data: pd.DataFrame) -> str:
    for column in OUTPUT_COLUMN_CANDIDATES:
        if column in data.columns:
            return column
    raise ValueError(
        f"Could not find Logistic Regression output column. Available columns: {list(data.columns)}"
    )


def threshold_slug(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def select_function_word_threshold(requested: str, log: EvaluationLog) -> float:
    if requested != "auto":
        return float(requested)
    summary_path = FUNCTION_WORD_DIR / "neural_filter_threshold_summary.csv"
    summary = pd.read_csv(summary_path)
    best = summary.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    threshold = float(best["threshold"])
    log.info(
        "Selected function-word-safe threshold "
        f"{threshold:.2f} from {summary_path} using highest F1={float(best['f1']):.4f}."
    )
    return threshold


def load_filtered(path: Path, log: EvaluationLog) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"sentence_id", "complex_sentence", "reference_simple_sentence"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    log.info(f"Loaded {path} with {len(data)} rows.")
    return data.sort_values("sentence_id").reset_index(drop=True)


def build_system_frames(
    args: argparse.Namespace, log: EvaluationLog
) -> tuple[pd.DataFrame, dict[str, str]]:
    selected_function_threshold = select_function_word_threshold(args.function_word_threshold, log)
    function_path = FUNCTION_WORD_DIR / (
        f"filtered_outputs_threshold_{threshold_slug(selected_function_threshold)}.csv"
    )
    content_070_path = CONTENT_QUALITY_DIR / "filtered_outputs_threshold_070.csv"
    content_080_path = CONTENT_QUALITY_DIR / "filtered_outputs_threshold_080.csv"

    function_data = load_filtered(function_path, log)
    content_070 = load_filtered(content_070_path, log)
    content_080 = load_filtered(content_080_path, log)

    lr_column = find_output_column(content_070)
    log.info(f"Logistic Regression output column used: {lr_column}")
    log.info("Filtered output column used for neural systems: neural_filtered_output")

    base = content_070[
        ["sentence_id", "complex_sentence", "reference_simple_sentence", lr_column]
    ].rename(columns={lr_column: "logistic_regression_only"})
    frames = [
        base,
        function_data[["sentence_id", "neural_filtered_output", "num_final_replacements"]].rename(
            columns={
                "neural_filtered_output": "function_word_safe_best",
                "num_final_replacements": "function_word_safe_best_replacements",
            }
        ),
        content_070[["sentence_id", "neural_filtered_output", "num_final_replacements"]].rename(
            columns={
                "neural_filtered_output": "content_quality_safe_threshold_070",
                "num_final_replacements": "content_quality_safe_threshold_070_replacements",
            }
        ),
        content_080[["sentence_id", "neural_filtered_output", "num_final_replacements"]].rename(
            columns={
                "neural_filtered_output": "content_quality_safe_threshold_080",
                "num_final_replacements": "content_quality_safe_threshold_080_replacements",
            }
        ),
    ]
    merged = frames[0]
    for frame in frames[1:]:
        before = len(merged)
        merged = merged.merge(frame, on="sentence_id", how="inner")
        if len(merged) != before:
            log.warning(f"Sentence-id alignment dropped {before - len(merged)} rows.")
    merged = merged.sort_values("sentence_id").reset_index(drop=True)

    replacement_counts = content_070[["sentence_id", "num_original_replacements"]].rename(
        columns={"num_original_replacements": "logistic_regression_only_replacements"}
    )
    merged = merged.merge(replacement_counts, on="sentence_id", how="left")
    thresholds = {
        "logistic_regression_only": "",
        "function_word_safe_best": f"{selected_function_threshold:.2f}",
        "content_quality_safe_threshold_070": "0.70",
        "content_quality_safe_threshold_080": "0.80",
    }
    return merged, thresholds


def per_sentence_rows(
    data: pd.DataFrame, system: str, output_column: str, repl_column: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        output = str(row[output_column])
        word_list = words(output)
        rows.append(
            {
                "sentence_id": row["sentence_id"],
                "complex_sentence": row["complex_sentence"],
                "reference_simple_sentence": row["reference_simple_sentence"],
                "system": system,
                "output_sentence": output,
                "token_count": len(tokens(output)),
                "sentence_length": len(output),
                "word_length": (sum(len(word) for word in word_list) / len(word_list))
                if word_list
                else 0.0,
                "source_token_overlap": token_f1(output, row["complex_sentence"]),
                "reference_token_overlap": token_f1(output, row["reference_simple_sentence"]),
                "num_replacements": int(row.get(repl_column, 0) or 0),
                "changed_from_source": normalize_text(output)
                != normalize_text(row["complex_sentence"]),
            }
        )
    return rows


def compute_bleu(outputs: list[str], references: list[str], log: EvaluationLog) -> float | None:
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(outputs, [references]).score)
    except Exception as exc:
        log.warning(f"BLEU unavailable: {exc}")
        return None


def compute_rouge(
    outputs: list[str], references: list[str], log: EvaluationLog
) -> dict[str, float | None]:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for output, reference in zip(outputs, references, strict=True):
            row_scores = scorer.score(reference, output)
            for key in scores:
                scores[key].append(row_scores[key].fmeasure)
        return {
            "rouge_1": float(sum(scores["rouge1"]) / len(scores["rouge1"])),
            "rouge_2": float(sum(scores["rouge2"]) / len(scores["rouge2"])),
            "rouge_l": float(sum(scores["rougeL"]) / len(scores["rougeL"])),
        }
    except Exception as exc:
        log.warning(f"ROUGE unavailable: {exc}")
        return {"rouge_1": None, "rouge_2": None, "rouge_l": None}


def compute_sari(
    sources: list[str], outputs: list[str], references: list[str], log: EvaluationLog
) -> float | None:
    try:
        from evaluate import load

        metric = load("sari")
        result = metric.compute(
            sources=sources,
            predictions=outputs,
            references=[[reference] for reference in references],
        )
        return float(result["sari"])
    except Exception as exc:
        log.warning(f"SARI unavailable; saved null. Reason: {exc}")
        return None


def compute_bertscore(
    outputs: list[str], references: list[str], log: EvaluationLog
) -> dict[str, float | None]:
    try:
        from bert_score import score

        precision, recall, f1 = score(
            cands=outputs,
            refs=references,
            lang="en",
            device="cpu",
            verbose=False,
        )
        return {
            "bertscore_precision": float(precision.mean()),
            "bertscore_recall": float(recall.mean()),
            "bertscore_f1": float(f1.mean()),
        }
    except Exception as exc:
        log.warning(f"BERTScore unavailable; saved null values. Reason: {exc}")
        return {"bertscore_precision": None, "bertscore_recall": None, "bertscore_f1": None}


def summarize_system(
    per_sentence: pd.DataFrame,
    sources: list[str],
    references: list[str],
    outputs: list[str],
    system: str,
    selected_threshold: str,
    log: EvaluationLog,
) -> dict[str, Any]:
    flesch, grade = readability_scores(outputs)
    rouge = compute_rouge(outputs, references, log)
    bert = compute_bertscore(outputs, references, log)
    row = {
        "system": system,
        "selected_threshold": selected_threshold,
        "num_sentences": int(len(outputs)),
        "avg_sentence_length": float(per_sentence["sentence_length"].mean()),
        "avg_token_count": float(per_sentence["token_count"].mean()),
        "avg_word_length": float(per_sentence["word_length"].mean()),
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": grade,
        "avg_replacements_per_sentence": float(per_sentence["num_replacements"].mean()),
        "percent_sentences_changed": float(per_sentence["changed_from_source"].mean() * 100),
        "source_token_overlap": float(per_sentence["source_token_overlap"].mean()),
        "reference_token_overlap": float(per_sentence["reference_token_overlap"].mean()),
        "bleu": compute_bleu(outputs, references, log),
        **rouge,
        "sari": compute_sari(sources, outputs, references, log),
        **bert,
        "entity_preservation_rate": entity_preservation_rate(sources, outputs, log),
        "number_preservation_rate": number_preservation_rate(sources, outputs),
    }
    log.info(f"Computed metrics for {system}.")
    return row


def write_qualitative_examples(data: pd.DataFrame, output_dir: Path) -> None:
    scored = data.copy()
    scored["lr_reference_overlap"] = scored.apply(
        lambda row: token_f1(row["logistic_regression_only"], row["reference_simple_sentence"]),
        axis=1,
    )
    scored["cq070_reference_overlap"] = scored.apply(
        lambda row: token_f1(
            row["content_quality_safe_threshold_070"], row["reference_simple_sentence"]
        ),
        axis=1,
    )
    scored["delta"] = scored["cq070_reference_overlap"] - scored["lr_reference_overlap"]

    best = scored.sort_values("delta", ascending=False).head(10)
    failures = scored.sort_values(
        ["delta", "cq070_reference_overlap"], ascending=[True, True]
    ).head(10)

    def format_examples(rows: pd.DataFrame, title: str) -> str:
        lines = [title, "=" * len(title), ""]
        for _, row in rows.iterrows():
            note = (
                f"Reference-token overlap changed by {row['delta']:+.3f} "
                f"(LR={row['lr_reference_overlap']:.3f}, "
                f"CQ70={row['cq070_reference_overlap']:.3f})."
            )
            lines.extend(
                [
                    f"Sentence ID: {row['sentence_id']}",
                    f"Complex: {row['complex_sentence']}",
                    f"Reference: {row['reference_simple_sentence']}",
                    f"Raw Logistic Regression: {row['logistic_regression_only']}",
                    f"Content-quality-safe 0.70: {row['content_quality_safe_threshold_070']}",
                    f"Automatic note: {note}",
                    "",
                ]
            )
        return "\n".join(lines)

    (output_dir / "qualitative_best_examples.txt").write_text(
        format_examples(best, "Best Content-Quality-Safe 0.70 Examples"),
        encoding="utf-8",
    )
    (output_dir / "qualitative_failure_examples.txt").write_text(
        format_examples(failures, "Failure or Conservative Content-Quality-Safe 0.70 Examples"),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["debug"], default="debug")
    parser.add_argument("--content_quality_thresholds", nargs="+", type=float, default=[0.70, 0.80])
    parser.add_argument("--function_word_threshold", default="auto")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    log = EvaluationLog()
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Log directory: {args.log_dir}")

    try:
        data, thresholds = build_system_frames(args, log)
    except Exception as exc:
        log.warning(f"Evaluation setup failed: {exc}")
        log.write(args.output_dir / "evaluation_log.txt")
        log.write(args.log_dir / "evaluation_log.txt")
        return 1

    systems = {
        "logistic_regression_only": (
            "logistic_regression_only",
            "logistic_regression_only_replacements",
        ),
        "function_word_safe_best": (
            "function_word_safe_best",
            "function_word_safe_best_replacements",
        ),
        "content_quality_safe_threshold_070": (
            "content_quality_safe_threshold_070",
            "content_quality_safe_threshold_070_replacements",
        ),
        "content_quality_safe_threshold_080": (
            "content_quality_safe_threshold_080",
            "content_quality_safe_threshold_080_replacements",
        ),
    }

    all_per_sentence: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sources = data["complex_sentence"].astype(str).tolist()
    references = data["reference_simple_sentence"].astype(str).tolist()
    for system, (output_column, repl_column) in systems.items():
        rows = per_sentence_rows(data, system, output_column, repl_column)
        all_per_sentence.extend(rows)
        per_sentence = pd.DataFrame(rows)
        outputs = data[output_column].astype(str).tolist()
        summary_rows.append(
            summarize_system(
                per_sentence,
                sources,
                references,
                outputs,
                system,
                thresholds[system],
                log,
            )
        )

    summary = pd.DataFrame(summary_rows).reindex(columns=SUMMARY_COLUMNS)
    per_sentence_df = pd.DataFrame(all_per_sentence)
    tradeoff = summary[
        [
            "system",
            "selected_threshold",
            "flesch_kincaid_grade",
            "sari",
            "bertscore_f1",
            "entity_preservation_rate",
            "number_preservation_rate",
            "avg_replacements_per_sentence",
            "percent_sentences_changed",
        ]
    ]

    summary.to_csv(args.output_dir / "final_filter_evaluation_summary.csv", index=False)
    (args.output_dir / "final_filter_evaluation_summary.json").write_text(
        json.dumps(summary.where(pd.notna(summary), None).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    per_sentence_df.to_csv(args.output_dir / "per_sentence_evaluation.csv", index=False)
    tradeoff.to_csv(args.output_dir / "tradeoff_table.csv", index=False)
    write_qualitative_examples(data, args.output_dir)
    log.info(
        "Metrics successfully computed: readability, lexical behavior, BLEU, ROUGE, "
        "entity/number preservation where available."
    )
    log.info("Any optional metric that is unavailable is recorded as null with a warning.")
    log.info(f"Selected function-word-safe threshold: {thresholds['function_word_safe_best']}")
    log.write(args.output_dir / "evaluation_log.txt")
    log.write(args.log_dir / "evaluation_log.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

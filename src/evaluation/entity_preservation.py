#!/usr/bin/env python3
# ruff: noqa: E501, UP017, UP030, UP032
# mypy: ignore-errors
"""Evaluate named-entity preservation in classical model simplification outputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import spacy


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "entity_preservation"

SOURCE_COLUMN_CANDIDATES = (
    "source_sentence",
    "original",
    "source",
    "complex",
    "complex_sentence",
    "input",
    "input_sentence",
)
OUTPUT_COLUMN_CANDIDATES = (
    "predicted_sentence",
    "simplified",
    "simplified_sentence",
    "candidate",
    "prediction",
    "output",
    "model_output",
)
REFERENCE_COLUMN_CANDIDATES = (
    "reference_sentence",
    "reference",
    "target",
    "target_sentence",
    "simple",
    "simple_sentence",
    "gold",
    "gold_sentence",
)

RUN_CANDIDATES = {
    ("Wikilarge", "Logistic Regression"): [
        "present/Predicted_sentences/logistic_regression_wikilarge_final_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_wikilarge_test_predictions.csv",
    ],
    ("Wikilarge", "SVM"): [
        "present/Predicted_sentences/svm_wikilarge_final_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_svm_wikilarge_full_20260609_test_predictions.csv",
    ],
    ("Wikilarge", "Random Forest"): [
        "present/Predicted_sentences/random_forest_wikilarge_final_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_random_forest_wikilarge_limit_10000_rf_small_10k_20260613_test_predictions.csv",
    ],
    ("Wikismall", "Logistic Regression"): [
        "present/Predicted_sentences/logistic_regression_wikismall_final_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_wikismall_test_predictions.csv",
    ],
    ("Wikismall", "SVM"): [
        "present/Predicted_sentences/svm_wikismall_final_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_svm_wikismall_full_20260609_test_predictions.csv",
    ],
    ("Wikismall", "Random Forest"): [
        "present/Predicted_sentences/random_forest_wikismall_final_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_random_forest_wikismall_rf_full_20260610_test_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_random_forest_wikismall_full_20260609_test_predictions.csv",
    ],
}

GLOB_FALLBACKS = {
    ("Wikilarge", "Logistic Regression"): [
        "present/Predicted_sentences/logistic_regression_wikilarge*.csv",
        "classical+logistical reg/outputs/classical_ml_wikilarge*test_predictions*.csv",
    ],
    ("Wikilarge", "SVM"): [
        "present/Predicted_sentences/svm_wikilarge*.csv",
        "classical+logistical reg/outputs/*svm*wikilarge*test_predictions*.csv",
    ],
    ("Wikilarge", "Random Forest"): [
        "present/Predicted_sentences/random_forest_wikilarge*.csv",
        "classical+logistical reg/outputs/*random_forest*wikilarge*test_predictions*.csv",
    ],
    ("Wikismall", "Logistic Regression"): [
        "present/Predicted_sentences/logistic_regression_wikismall*.csv",
        "classical+logistical reg/outputs/classical_ml_wikismall*test_predictions*.csv",
    ],
    ("Wikismall", "SVM"): [
        "present/Predicted_sentences/svm_wikismall*.csv",
        "classical+logistical reg/outputs/*svm*wikismall*test_predictions*.csv",
    ],
    ("Wikismall", "Random Forest"): [
        "present/Predicted_sentences/random_forest_wikismall*.csv",
        "classical+logistical reg/outputs/*random_forest*wikismall*test_predictions*.csv",
    ],
}

SUMMARY_FIELDS = [
    "dataset_name",
    "model_name",
    "total_sentence_pairs",
    "sentences_with_entities",
    "total_entities_in_original",
    "strict_preserved_entities",
    "strict_lost_entities",
    "strict_entity_preservation_rate",
    "strict_entity_loss_rate",
    "type_level_preserved_entities",
    "type_level_lost_entities",
    "type_level_entity_preservation_rate",
    "type_level_entity_loss_rate",
    "reference_strict_preserved_entities",
    "reference_strict_lost_entities",
    "reference_strict_entity_preservation_rate",
    "reference_strict_entity_loss_rate",
    "reference_type_level_preserved_entities",
    "reference_type_level_lost_entities",
    "reference_type_level_entity_preservation_rate",
    "reference_type_level_entity_loss_rate",
]

TYPE_FIELDS = [
    "dataset_name",
    "model_name",
    "entity_type",
    "total_entities",
    "strict_preserved_entities",
    "strict_lost_entities",
    "strict_preservation_rate",
    "strict_loss_rate",
    "type_level_preserved_entities",
    "type_level_lost_entities",
    "type_level_preservation_rate",
    "type_level_loss_rate",
    "reference_strict_preserved_entities",
    "reference_strict_lost_entities",
    "reference_strict_preservation_rate",
    "reference_strict_loss_rate",
    "reference_type_level_preserved_entities",
    "reference_type_level_lost_entities",
    "reference_type_level_preservation_rate",
    "reference_type_level_loss_rate",
]

EXAMPLE_FIELDS = [
    "dataset_name",
    "model_name",
    "row_id",
    "original_sentence",
    "reference_sentence",
    "model_output",
    "original_entities_identified",
    "reference_entities_identified",
    "output_entities_identified",
    "strict_preserved_entities",
    "strict_lost_entities",
    "type_level_preserved_entity_types",
    "type_level_lost_entity_types",
    "reference_strict_preserved_entities",
    "reference_strict_lost_entities",
    "reference_type_level_preserved_entity_types",
    "reference_type_level_lost_entity_types",
    "strict_comparison",
    "type_level_comparison",
]


def normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def normalize_entity_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(".,;:!?\"'`()[]{}")


def find_prediction_file(dataset_name: str, model_name: str) -> Path | None:
    key = (dataset_name, model_name)

    for relative_path in RUN_CANDIDATES[key]:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            return path

    for pattern in GLOB_FALLBACKS[key]:
        matches = sorted(
            PROJECT_ROOT.glob(pattern),
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        )
        for path in matches:
            lower_name = path.name.lower()
            if "limit_20" in lower_name or "smoke" in lower_name or "verification" in lower_name:
                continue
            if model_name == "Logistic Regression" and (
                "svm" in lower_name or "random_forest" in lower_name or "_rf_" in lower_name
            ):
                continue
            return path

    return None


def detect_columns(fieldnames: list[str]) -> tuple[str, str]:
    normalized = {normalize_header(field): field for field in fieldnames}
    source_column = next(
        (normalized[name] for name in SOURCE_COLUMN_CANDIDATES if name in normalized),
        None,
    )
    output_column = next(
        (normalized[name] for name in OUTPUT_COLUMN_CANDIDATES if name in normalized),
        None,
    )

    if not source_column or not output_column:
        raise ValueError(
            f"Could not detect source/output columns. Available columns: {', '.join(fieldnames)}"
        )

    return source_column, output_column


def detect_reference_column(fieldnames: list[str]) -> str | None:
    normalized = {normalize_header(field): field for field in fieldnames}
    return next(
        (normalized[name] for name in REFERENCE_COLUMN_CANDIDATES if name in normalized),
        None,
    )


def load_rows(path: Path) -> tuple[list[dict[str, Any]], str, str, str | None]:
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            source_column, output_column = detect_columns(fieldnames)
            reference_column = detect_reference_column(fieldnames)
            return list(reader), source_column, output_column, reference_column

    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("predictions") or data.get("rows") or data.get("data") or []
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError(f"Unsupported JSON prediction format in {path}")
        fieldnames = sorted({key for row in data for key in row})
        source_column, output_column = detect_columns(fieldnames)
        reference_column = detect_reference_column(fieldnames)
        return data, source_column, output_column, reference_column

    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        fieldnames = sorted({key for row in rows for key in row})
        source_column, output_column = detect_columns(fieldnames)
        reference_column = detect_reference_column(fieldnames)
        return rows, source_column, output_column, reference_column

    raise ValueError(f"Unsupported prediction file type: {path}")


def load_ner_model() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise SystemExit(
            "Could not load spaCy model 'en_core_web_sm'. "
            "Install project dependencies first, for example with: uv sync"
        ) from exc


def extract_entity_records(doc: Any) -> list[dict[str, Any]]:
    return [
        {
            "text": entity.text,
            "normalized_text": normalize_entity_text(entity.text),
            "entity_type": entity.label_,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
        }
        for entity in doc.ents
        if normalize_entity_text(entity.text)
    ]


def entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return entity["normalized_text"], entity["entity_type"]


def overlap_count(original_counts: Counter[Any], output_counts: Counter[Any]) -> int:
    return sum(min(count, output_counts[item]) for item, count in original_counts.items())


def counter_preserved_items(
    original_counts: Counter[Any], output_counts: Counter[Any]
) -> list[Any]:
    preserved = []
    for item in sorted(original_counts):
        preserved.extend([item] * min(original_counts[item], output_counts[item]))
    return preserved


def counter_lost_items(original_counts: Counter[Any], output_counts: Counter[Any]) -> list[Any]:
    lost = []
    for item in sorted(original_counts):
        missing_count = original_counts[item] - output_counts[item]
        if missing_count > 0:
            lost.extend([item] * missing_count)
    return lost


def rate(part: int, total: int) -> float:
    return part / total if total else 0.0


def format_entity_records(entities: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{entity['text']} [{entity['entity_type']}] normalized={entity['normalized_text']}"
        for entity in entities
    )


def format_entity_keys(items: list[tuple[str, str]]) -> str:
    return "; ".join(f"{text} [{entity_type}]" for text, entity_type in items)


def format_entity_types(items: list[str]) -> str:
    return "; ".join(items)


def build_example_row(
    dataset_name: str,
    model_name: str,
    row_id: int,
    original_sentence: str,
    reference_sentence: str,
    model_output: str,
    original_entities: list[dict[str, Any]],
    reference_entities: list[dict[str, Any]],
    output_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    original_strict_counts = Counter(entity_key(entity) for entity in original_entities)
    reference_strict_counts = Counter(entity_key(entity) for entity in reference_entities)
    output_strict_counts = Counter(entity_key(entity) for entity in output_entities)
    original_type_counts = Counter(entity["entity_type"] for entity in original_entities)
    reference_type_counts = Counter(entity["entity_type"] for entity in reference_entities)
    output_type_counts = Counter(entity["entity_type"] for entity in output_entities)

    strict_preserved = counter_preserved_items(original_strict_counts, output_strict_counts)
    strict_lost = counter_lost_items(original_strict_counts, output_strict_counts)
    type_level_preserved = counter_preserved_items(original_type_counts, output_type_counts)
    type_level_lost = counter_lost_items(original_type_counts, output_type_counts)
    reference_strict_preserved = counter_preserved_items(
        original_strict_counts,
        reference_strict_counts,
    )
    reference_strict_lost = counter_lost_items(original_strict_counts, reference_strict_counts)
    reference_type_level_preserved = counter_preserved_items(
        original_type_counts,
        reference_type_counts,
    )
    reference_type_level_lost = counter_lost_items(original_type_counts, reference_type_counts)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "row_id": row_id,
        "original_sentence": original_sentence,
        "reference_sentence": reference_sentence,
        "model_output": model_output,
        "original_entities_identified": format_entity_records(original_entities),
        "reference_entities_identified": format_entity_records(reference_entities),
        "output_entities_identified": format_entity_records(output_entities),
        "strict_preserved_entities": format_entity_keys(strict_preserved),
        "strict_lost_entities": format_entity_keys(strict_lost),
        "type_level_preserved_entity_types": format_entity_types(type_level_preserved),
        "type_level_lost_entity_types": format_entity_types(type_level_lost),
        "reference_strict_preserved_entities": format_entity_keys(reference_strict_preserved),
        "reference_strict_lost_entities": format_entity_keys(reference_strict_lost),
        "reference_type_level_preserved_entity_types": format_entity_types(
            reference_type_level_preserved
        ),
        "reference_type_level_lost_entity_types": format_entity_types(reference_type_level_lost),
        "strict_comparison": (
            "Strict match counts an entity as preserved only when normalized text and "
            "entity type both appear in the model output or reference sentence."
        ),
        "type_level_comparison": (
            "Type-level match counts an entity slot as preserved when the same entity "
            "type appears in the model output or reference sentence, even if the text changed."
        ),
    }


def evaluate_run(
    dataset_name: str,
    model_name: str,
    rows: list[dict[str, Any]],
    source_column: str,
    output_column: str,
    reference_column: str | None,
    nlp: spacy.language.Language,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    total_sentence_pairs = len(rows)
    sentences_with_entities = 0
    total_entities = 0
    strict_preserved = 0
    type_level_preserved = 0
    reference_strict_preserved = 0
    reference_type_level_preserved = 0
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total_entities": 0,
            "strict_preserved_entities": 0,
            "type_level_preserved_entities": 0,
            "reference_strict_preserved_entities": 0,
            "reference_type_level_preserved_entities": 0,
        }
    )
    example_candidates = []

    source_texts = [str(row.get(source_column, "") or "") for row in rows]
    output_texts = [str(row.get(output_column, "") or "") for row in rows]
    reference_texts = [
        str(row.get(reference_column, "") or "") if reference_column else "" for row in rows
    ]

    for row_id, (source_doc, output_doc, reference_doc) in enumerate(
        zip(
            nlp.pipe(source_texts),
            nlp.pipe(output_texts),
            nlp.pipe(reference_texts),
            strict=True,
        ),
        start=1,
    ):
        original_entity_records = extract_entity_records(source_doc)
        original_entities = [entity_key(entity) for entity in original_entity_records]
        if not original_entities:
            continue

        reference_entity_records = extract_entity_records(reference_doc)
        reference_entities = [entity_key(entity) for entity in reference_entity_records]
        output_entity_records = extract_entity_records(output_doc)
        output_entities = [entity_key(entity) for entity in output_entity_records]
        original_strict_counts = Counter(original_entities)
        reference_strict_counts = Counter(reference_entities)
        output_strict_counts = Counter(output_entities)
        original_type_counts = Counter(entity_type for _, entity_type in original_entities)
        reference_type_counts = Counter(entity_type for _, entity_type in reference_entities)
        output_type_counts = Counter(entity_type for _, entity_type in output_entities)
        example_candidates.append(
            build_example_row(
                dataset_name,
                model_name,
                row_id,
                source_texts[row_id - 1],
                reference_texts[row_id - 1],
                output_texts[row_id - 1],
                original_entity_records,
                reference_entity_records,
                output_entity_records,
            )
        )

        sentences_with_entities += 1
        total_entities += len(original_entities)
        strict_preserved += overlap_count(original_strict_counts, output_strict_counts)
        type_level_preserved += overlap_count(original_type_counts, output_type_counts)
        reference_strict_preserved += overlap_count(original_strict_counts, reference_strict_counts)
        reference_type_level_preserved += overlap_count(original_type_counts, reference_type_counts)

        for entity_type, count in original_type_counts.items():
            strict_type_original = Counter(
                entity for entity in original_entities if entity[1] == entity_type
            )
            strict_type_output = Counter(
                entity for entity in output_entities if entity[1] == entity_type
            )
            strict_type_reference = Counter(
                entity for entity in reference_entities if entity[1] == entity_type
            )
            strict_type_preserved = overlap_count(strict_type_original, strict_type_output)
            type_preserved = min(count, output_type_counts[entity_type])
            reference_strict_type_preserved = overlap_count(
                strict_type_original,
                strict_type_reference,
            )
            reference_type_preserved = min(count, reference_type_counts[entity_type])

            by_type[entity_type]["total_entities"] += count
            by_type[entity_type]["strict_preserved_entities"] += strict_type_preserved
            by_type[entity_type]["type_level_preserved_entities"] += type_preserved
            by_type[entity_type]["reference_strict_preserved_entities"] += (
                reference_strict_type_preserved
            )
            by_type[entity_type]["reference_type_level_preserved_entities"] += (
                reference_type_preserved
            )

    strict_lost = total_entities - strict_preserved
    type_level_lost = total_entities - type_level_preserved
    reference_strict_lost = total_entities - reference_strict_preserved
    reference_type_level_lost = total_entities - reference_type_level_preserved
    summary = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "total_sentence_pairs": total_sentence_pairs,
        "sentences_with_entities": sentences_with_entities,
        "total_entities_in_original": total_entities,
        "strict_preserved_entities": strict_preserved,
        "strict_lost_entities": strict_lost,
        "strict_entity_preservation_rate": rate(strict_preserved, total_entities),
        "strict_entity_loss_rate": rate(strict_lost, total_entities),
        "type_level_preserved_entities": type_level_preserved,
        "type_level_lost_entities": type_level_lost,
        "type_level_entity_preservation_rate": rate(type_level_preserved, total_entities),
        "type_level_entity_loss_rate": rate(type_level_lost, total_entities),
        "reference_strict_preserved_entities": reference_strict_preserved,
        "reference_strict_lost_entities": reference_strict_lost,
        "reference_strict_entity_preservation_rate": rate(
            reference_strict_preserved,
            total_entities,
        ),
        "reference_strict_entity_loss_rate": rate(reference_strict_lost, total_entities),
        "reference_type_level_preserved_entities": reference_type_level_preserved,
        "reference_type_level_lost_entities": reference_type_level_lost,
        "reference_type_level_entity_preservation_rate": rate(
            reference_type_level_preserved,
            total_entities,
        ),
        "reference_type_level_entity_loss_rate": rate(
            reference_type_level_lost,
            total_entities,
        ),
    }

    type_rows = []
    for entity_type in sorted(by_type):
        row = by_type[entity_type]
        type_total = row["total_entities"]
        strict_type_lost = type_total - row["strict_preserved_entities"]
        type_level_type_lost = type_total - row["type_level_preserved_entities"]
        reference_strict_type_lost = type_total - row["reference_strict_preserved_entities"]
        reference_type_level_type_lost = type_total - row["reference_type_level_preserved_entities"]
        type_rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "entity_type": entity_type,
                "total_entities": type_total,
                "strict_preserved_entities": row["strict_preserved_entities"],
                "strict_lost_entities": strict_type_lost,
                "strict_preservation_rate": rate(row["strict_preserved_entities"], type_total),
                "strict_loss_rate": rate(strict_type_lost, type_total),
                "type_level_preserved_entities": row["type_level_preserved_entities"],
                "type_level_lost_entities": type_level_type_lost,
                "type_level_preservation_rate": rate(
                    row["type_level_preserved_entities"], type_total
                ),
                "type_level_loss_rate": rate(type_level_type_lost, type_total),
                "reference_strict_preserved_entities": row["reference_strict_preserved_entities"],
                "reference_strict_lost_entities": reference_strict_type_lost,
                "reference_strict_preservation_rate": rate(
                    row["reference_strict_preserved_entities"],
                    type_total,
                ),
                "reference_strict_loss_rate": rate(reference_strict_type_lost, type_total),
                "reference_type_level_preserved_entities": row[
                    "reference_type_level_preserved_entities"
                ],
                "reference_type_level_lost_entities": reference_type_level_type_lost,
                "reference_type_level_preservation_rate": rate(
                    row["reference_type_level_preserved_entities"],
                    type_total,
                ),
                "reference_type_level_loss_rate": rate(
                    reference_type_level_type_lost,
                    type_total,
                ),
            }
        )

    return summary, type_rows, example_candidates


def format_rate(value: float) -> str:
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


def markdown_value(row: dict[str, Any], field: str) -> Any:
    value = row[field]
    return format_rate(value) if field.endswith("_rate") else value


def write_markdown_table(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, bool]],
) -> None:
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(label for _, label, _ in columns) + " |",
        "|" + "|".join("---:" if numeric else "---" for _, _, numeric in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(markdown_value(row, field)) for field, _, _ in columns) + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def worksheet_xml(rows: list[list[Any]], column_widths: list[int]) -> str:
    last_column = column_letter(max(len(row) for row in rows)) if rows else "A"
    cols = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(column_widths, start=1)
    )
    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            cell_ref = f"{column_letter(col_index)}{row_index}"
            cell_value = clean_cell(value)
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">'
                f"{escape(cell_value)}</t></is></c>"
            )
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<cols>{cols}</cols>"
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        f'<autoFilter ref="A1:{last_column}1"/>'
        "</worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets>"
        "</workbook>"
    )


def workbook_relationships(sheet_count: int) -> str:
    relationships = "".join(
        '<Relationship Id="rId{0}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet{0}.xml"/>'.format(index)
        for index in range(1, sheet_count + 1)
    )
    relationships += (
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )


def content_types(sheet_count: int) -> str:
    sheets = "".join(
        '<Override PartName="/xl/worksheets/sheet{0}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(
            index
        )
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{sheets}</Types>"
    )


def save_workbook(
    path: Path,
    summary_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> None:
    result_rows = [
        SUMMARY_FIELDS,
        *[[row[field] for field in SUMMARY_FIELDS] for row in summary_rows],
    ]
    random_example_rows = [
        EXAMPLE_FIELDS,
        *[[row[field] for field in EXAMPLE_FIELDS] for row in example_rows],
    ]
    sheets = {
        "Results": (
            result_rows,
            [
                16,
                24,
                18,
                24,
                22,
                22,
                16,
                24,
                18,
                26,
                20,
                30,
                22,
                28,
                22,
                34,
                28,
                34,
                28,
                40,
                34,
            ],
        ),
        "Random Examples": (
            random_example_rows,
            [16, 24, 10, 70, 70, 70, 70, 70, 70, 45, 45, 28, 28, 45, 45, 34, 34, 55, 55],
        ),
    }
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types(len(sheets)))
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
            'Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
            'Target="docProps/app.xml"/>'
            "</Relationships>",
        )
        archive.writestr("xl/workbook.xml", workbook_xml(list(sheets)))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships(len(sheets)))
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>",
        )
        archive.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:creator>Entity Preservation Analysis</dc:creator>"
            "<dc:title>Entity Preservation Results</dc:title>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
            "</cp:coreProperties>",
        )
        archive.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>Entity Preservation Analysis</Application>"
            f'<TitlesOfParts><vt:vector size="{len(sheets)}" baseType="lpstr">'
            + "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheets)
            + "</vt:vector></TitlesOfParts>"
            "</Properties>",
        )
        for index, (rows, widths) in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml(rows, widths))


def save_results(
    summary_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(RESULTS_DIR / "entity_preservation_results.csv", summary_rows, SUMMARY_FIELDS)
    write_json(RESULTS_DIR / "entity_preservation_results.json", summary_rows)
    save_workbook(RESULTS_DIR / "entity_preservation_results.xlsx", summary_rows, example_rows)
    write_markdown_table(
        RESULTS_DIR / "entity_preservation_results.md",
        "Entity Preservation Results",
        summary_rows,
        [
            ("dataset_name", "Dataset", False),
            ("model_name", "Model", False),
            ("total_sentence_pairs", "Sentence Pairs", True),
            ("sentences_with_entities", "Sentences with Entities", True),
            ("total_entities_in_original", "Total Entities", True),
            ("strict_preserved_entities", "Strict Preserved", True),
            ("strict_lost_entities", "Strict Lost", True),
            ("strict_entity_preservation_rate", "Strict Preservation Rate", True),
            ("strict_entity_loss_rate", "Strict Loss Rate", True),
            ("type_level_preserved_entities", "Type-Level Preserved", True),
            ("type_level_lost_entities", "Type-Level Lost", True),
            ("type_level_entity_preservation_rate", "Type-Level Preservation Rate", True),
            ("type_level_entity_loss_rate", "Type-Level Loss Rate", True),
            ("reference_strict_preserved_entities", "Reference Strict Preserved", True),
            ("reference_strict_lost_entities", "Reference Strict Lost", True),
            (
                "reference_strict_entity_preservation_rate",
                "Reference Strict Preservation Rate",
                True,
            ),
            ("reference_strict_entity_loss_rate", "Reference Strict Loss Rate", True),
            ("reference_type_level_preserved_entities", "Reference Type-Level Preserved", True),
            ("reference_type_level_lost_entities", "Reference Type-Level Lost", True),
            (
                "reference_type_level_entity_preservation_rate",
                "Reference Type-Level Preservation Rate",
                True,
            ),
            ("reference_type_level_entity_loss_rate", "Reference Type-Level Loss Rate", True),
        ],
    )

    write_csv(RESULTS_DIR / "entity_preservation_by_type.csv", type_rows, TYPE_FIELDS)
    write_json(RESULTS_DIR / "entity_preservation_by_type.json", type_rows)
    write_markdown_table(
        RESULTS_DIR / "entity_preservation_by_type.md",
        "Entity Preservation by Type",
        type_rows,
        [
            ("dataset_name", "Dataset", False),
            ("model_name", "Model", False),
            ("entity_type", "Entity Type", False),
            ("total_entities", "Total", True),
            ("strict_preserved_entities", "Strict Preserved", True),
            ("strict_lost_entities", "Strict Lost", True),
            ("strict_preservation_rate", "Strict Rate", True),
            ("strict_loss_rate", "Strict Loss Rate", True),
            ("type_level_preserved_entities", "Type-Level Preserved", True),
            ("type_level_lost_entities", "Type-Level Lost", True),
            ("type_level_preservation_rate", "Type-Level Rate", True),
            ("type_level_loss_rate", "Type-Level Loss Rate", True),
            ("reference_strict_preserved_entities", "Reference Strict Preserved", True),
            ("reference_strict_lost_entities", "Reference Strict Lost", True),
            ("reference_strict_preservation_rate", "Reference Strict Rate", True),
            ("reference_strict_loss_rate", "Reference Strict Loss Rate", True),
            ("reference_type_level_preserved_entities", "Reference Type-Level Preserved", True),
            ("reference_type_level_lost_entities", "Reference Type-Level Lost", True),
            ("reference_type_level_preservation_rate", "Reference Type-Level Rate", True),
            ("reference_type_level_loss_rate", "Reference Type-Level Loss Rate", True),
        ],
    )


def select_random_examples(
    example_candidates: list[dict[str, Any]], count: int = 20
) -> list[dict[str, Any]]:
    if len(example_candidates) <= count:
        return example_candidates
    return random.Random(42).sample(example_candidates, count)


def evaluate_all(
    input_path: Path | None = None,
    dataset_name: str = "Dataset",
    model_name: str = "Model",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nlp = load_ner_model()
    summary_rows = []
    type_rows = []
    example_candidates = []

    if input_path is not None:
        rows, source_column, output_column, reference_column = load_rows(input_path)
        if not rows:
            raise ValueError(f"No rows found in prediction file: {input_path}")
        summary, run_type_rows, run_example_candidates = evaluate_run(
            dataset_name,
            model_name,
            rows,
            source_column,
            output_column,
            reference_column,
            nlp,
        )
        return [summary], run_type_rows, select_random_examples(run_example_candidates)

    for dataset_name, model_name in RUN_CANDIDATES:
        prediction_file = find_prediction_file(dataset_name, model_name)
        if prediction_file is None:
            print(f"Warning: no prediction file found for {dataset_name} {model_name}; skipping.")
            continue

        rows, source_column, output_column, reference_column = load_rows(prediction_file)
        print(
            f"Evaluating {dataset_name} / {model_name}: {prediction_file.relative_to(PROJECT_ROOT)}"
        )
        summary, run_type_rows, run_example_candidates = evaluate_run(
            dataset_name,
            model_name,
            rows,
            source_column,
            output_column,
            reference_column,
            nlp,
        )
        summary_rows.append(summary)
        type_rows.extend(run_type_rows)
        example_candidates.extend(run_example_candidates)

    if not summary_rows:
        raise FileNotFoundError("No usable prediction files were found.")

    return summary_rows, type_rows, select_random_examples(example_candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Evaluate one prediction CSV/JSON/JSONL/TSV file.")
    parser.add_argument("--dataset-name", default="Dataset", help="Dataset label used when --input is provided.")
    parser.add_argument("--model-name", default="Model", help="Model label used when --input is provided.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows, type_rows, example_rows = evaluate_all(
        input_path=args.input,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    save_results(summary_rows, type_rows, example_rows)

    print(f"\nSaved entity preservation results to {RESULTS_DIR.relative_to(PROJECT_ROOT)}")
    for path in sorted(RESULTS_DIR.glob("entity_preservation*")):
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

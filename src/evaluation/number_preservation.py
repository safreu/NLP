#!/usr/bin/env python3
# ruff: noqa: E501, SIM105, UP017, UP030, UP032
# mypy: ignore-errors
"""Evaluate number preservation in classical model simplification outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "number_preservation"

SOURCE_COLUMN_CANDIDATES = (
    "original",
    "source",
    "source_sentence",
    "complex",
    "complex_sentence",
    "input",
    "input_sentence",
)
OUTPUT_COLUMN_CANDIDATES = (
    "simplified",
    "simplified_sentence",
    "candidate",
    "prediction",
    "predicted_sentence",
    "output",
    "model_output",
)
REFERENCE_COLUMN_CANDIDATES = (
    "reference",
    "reference_sentence",
    "target",
    "target_sentence",
    "simple",
    "simple_sentence",
    "gold",
    "gold_sentence",
)

RUN_CANDIDATES = {
    ("Logistic Regression", "Wikilarge"): [
        "classical+logistical reg/outputs/classical_ml_wikilarge_test_predictions.csv",
        "present/Predicted_sentences/logistic_regression_wikilarge_final_predictions.csv",
    ],
    ("Logistic Regression", "Wikismall"): [
        "classical+logistical reg/outputs/classical_ml_wikismall_test_predictions.csv",
        "present/Predicted_sentences/logistic_regression_wikismall_final_predictions.csv",
    ],
    ("SVM", "Wikilarge"): [
        "classical+logistical reg/outputs/classical_ml_svm_wikilarge_full_20260609_test_predictions.csv",
        "present/Predicted_sentences/svm_wikilarge_final_predictions.csv",
    ],
    ("SVM", "Wikismall"): [
        "classical+logistical reg/outputs/classical_ml_svm_wikismall_full_20260609_test_predictions.csv",
        "present/Predicted_sentences/svm_wikismall_final_predictions.csv",
    ],
    ("Random Forest", "Wikilarge"): [
        "classical+logistical reg/outputs/classical_ml_random_forest_wikilarge_limit_10000_rf_small_10k_20260613_test_predictions.csv",
        "present/Predicted_sentences/random_forest_wikilarge_final_predictions.csv",
    ],
    ("Random Forest", "Wikismall"): [
        "classical+logistical reg/outputs/classical_ml_random_forest_wikismall_rf_full_20260610_test_predictions.csv",
        "classical+logistical reg/outputs/classical_ml_random_forest_wikismall_full_20260609_test_predictions.csv",
        "present/Predicted_sentences/random_forest_wikismall_final_predictions.csv",
    ],
}

GLOB_FALLBACKS = {
    ("Logistic Regression", "Wikilarge"): [
        "classical+logistical reg/outputs/classical_ml_wikilarge*test_predictions*.csv",
        "present/Predicted_sentences/logistic_regression_wikilarge*.csv",
    ],
    ("Logistic Regression", "Wikismall"): [
        "classical+logistical reg/outputs/classical_ml_wikismall_test_predictions.csv",
        "present/Predicted_sentences/logistic_regression_wikismall*.csv",
    ],
    ("SVM", "Wikilarge"): [
        "classical+logistical reg/outputs/*svm*wikilarge*test_predictions*.csv",
        "present/Predicted_sentences/svm_wikilarge*.csv",
    ],
    ("SVM", "Wikismall"): [
        "classical+logistical reg/outputs/*svm*wikismall_full*test_predictions*.csv",
        "present/Predicted_sentences/svm_wikismall*.csv",
    ],
    ("Random Forest", "Wikilarge"): [
        "classical+logistical reg/outputs/*random_forest*wikilarge*test_predictions*.csv",
        "present/Predicted_sentences/random_forest_wikilarge*.csv",
    ],
    ("Random Forest", "Wikismall"): [
        "classical+logistical reg/outputs/*random_forest*wikismall*rf_full*test_predictions*.csv",
        "classical+logistical reg/outputs/*random_forest*wikismall_full*test_predictions*.csv",
        "present/Predicted_sentences/random_forest_wikismall*.csv",
    ],
}

NUMBER_PATTERN = re.compile(
    r"(?<!\w)"
    r"[-+]?"
    r"(?:\d{1,3}(?:,\s*\d{3})+|\d+)"
    r"(?:\s*\.\s*\d+)?"
    r"\s*%?"
    r"(?!\w)"
)


def normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def normalize_number(number: str) -> str:
    number = number.strip()
    has_percent = number.replace(" ", "").endswith("%")
    core = number.rstrip("%") if has_percent else number
    core = core.replace(",", "")
    core = re.sub(r"\s+", "", core)

    if "." in core:
        try:
            core = str(float(core)).rstrip("0").rstrip(".")
        except ValueError:
            pass

    return f"{core}%" if has_percent else core


def extract_numbers(text: Any) -> list[str]:
    if text is None:
        return []
    return [normalize_number(match.group(0)) for match in NUMBER_PATTERN.finditer(str(text))]


def flexible_number_pattern(number: str) -> re.Pattern[str]:
    escaped_parts = []
    for character in number:
        if character == ".":
            escaped_parts.append(r"\s*\.\s*")
        elif character == "%":
            escaped_parts.append(r"\s*%\s*")
        else:
            escaped_parts.append(re.escape(character) + r"\s*")
    return re.compile(r"(?<!\w)" + "".join(escaped_parts).rstrip(r"\s*") + r"(?!\w)")


def flexible_occurrence_count(number: str, text: Any) -> int:
    if text is None:
        return 0
    return len(flexible_number_pattern(number).findall(str(text)))


def compare_numbers(
    original_numbers: list[str],
    output_numbers: list[str],
    output_text: Any = "",
) -> dict[str, Any]:
    original_counts = Counter(original_numbers)
    output_counts = Counter(output_numbers)
    preserved_numbers = 0
    lost_values: list[str] = []
    for number, count in original_counts.items():
        available_count = max(output_counts[number], flexible_occurrence_count(number, output_text))
        preserved_count = min(count, available_count)
        preserved_numbers += preserved_count
        missing_count = count - preserved_count
        if missing_count > 0:
            lost_values.extend([number] * missing_count)

    return {
        "preserved_numbers": preserved_numbers,
        "lost_numbers": len(lost_values),
        "lost_number_values": lost_values,
        "has_number_loss": bool(lost_values),
    }


def find_prediction_file(model_name: str, dataset_name: str) -> tuple[Path | None, list[Path]]:
    checked: list[Path] = []
    key = (model_name, dataset_name)

    for relative_path in RUN_CANDIDATES[key]:
        path = PROJECT_ROOT / relative_path
        checked.append(path)
        if path.exists():
            return path, checked

    for pattern in GLOB_FALLBACKS[key]:
        matches = sorted(
            PROJECT_ROOT.glob(pattern),
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        )
        checked.extend(matches)
        for path in matches:
            lower_name = path.name.lower()
            if "limit_20" in lower_name or "smoke" in lower_name or "verification" in lower_name:
                continue
            if model_name == "Logistic Regression" and (
                "svm" in lower_name or "random_forest" in lower_name or "_rf_" in lower_name
            ):
                continue
            return path, checked

    return None, checked


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


def evaluate_model(
    model_name: str,
    dataset_name: str,
    rows: list[dict[str, Any]],
    source_column: str,
    output_column: str,
) -> dict[str, Any]:
    total_pairs = len(rows)
    sentences_with_numbers = 0
    total_numbers = 0
    preserved_numbers = 0

    for row in rows:
        original_numbers = extract_numbers(row.get(source_column, ""))
        if not original_numbers:
            continue

        sentences_with_numbers += 1
        simplified_text = row.get(output_column, "")
        simplified_numbers = extract_numbers(simplified_text)
        comparison = compare_numbers(original_numbers, simplified_numbers, simplified_text)

        total_numbers += len(original_numbers)
        preserved_numbers += comparison["preserved_numbers"]

    lost_numbers = total_numbers - preserved_numbers
    preservation_rate = preserved_numbers / total_numbers if total_numbers else 0.0
    loss_rate = lost_numbers / total_numbers if total_numbers else 0.0

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "total_sentence_pairs": total_pairs,
        "sentences_with_numbers": sentences_with_numbers,
        "total_numbers_in_original": total_numbers,
        "preserved_numbers": preserved_numbers,
        "lost_numbers": lost_numbers,
        "number_preservation_rate": preservation_rate,
        "number_loss_rate": loss_rate,
    }


def format_rate(value: float) -> str:
    return f"{value:.4f}"


def save_results(results: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "number_preservation_results.csv"
    json_path = RESULTS_DIR / "number_preservation_results.json"
    markdown_path = RESULTS_DIR / "number_preservation_results.md"
    workbook_path = RESULTS_DIR / "number_preservation_results.xlsx"
    sentence_details = build_sentence_details()

    save_combined_csv(csv_path, results, sentence_details)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": results,
                "sentence_details": sentence_details,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    lines = [
        "# Number Preservation Results",
        "",
        "| Model | Dataset | Sentence Pairs | Sentences with Numbers | Total Numbers | Preserved Numbers | Lost Numbers | Preservation Rate | Loss Rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {model_name} | {dataset} | {total_sentence_pairs} | {sentences_with_numbers} | "
            "{total_numbers_in_original} | {preserved_numbers} | {lost_numbers} | "
            "{number_preservation_rate} | {number_loss_rate} |".format(
                **{
                    **row,
                    "number_preservation_rate": format_rate(row["number_preservation_rate"]),
                    "number_loss_rate": format_rate(row["number_loss_rate"]),
                }
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    examples = build_wikismall_examples()
    save_workbook(workbook_path, results, sentence_details, examples)


def save_combined_csv(
    path: Path,
    results: list[dict[str, Any]],
    sentence_details: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "section",
        "model_name",
        "dataset",
        "row_id",
        "total_sentence_pairs",
        "sentences_with_numbers",
        "total_numbers_in_original",
        "preserved_numbers",
        "lost_numbers",
        "number_preservation_rate",
        "number_loss_rate",
        "original_numbers",
        "output_numbers",
        "lost_number_values",
        "has_number_loss",
        "original_sentence",
        "reference_sentence",
        "model_output",
    ]
    rows: list[dict[str, Any]] = []

    for result in results:
        rows.append(
            {
                "section": "summary",
                **result,
            }
        )

    for detail in sentence_details:
        rows.append(
            {
                "section": "sentence_detail",
                **detail,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_sentence_details() -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []

    for model_name, dataset_name in RUN_CANDIDATES:
        prediction_file, _ = find_prediction_file(model_name, dataset_name)
        if prediction_file is None:
            continue

        try:
            rows, source_column, output_column, reference_column = load_rows(prediction_file)
        except Exception as exc:
            print(
                f"Warning: could not build sentence details for {model_name} {dataset_name}: {exc}"
            )
            continue

        for index, row in enumerate(rows, start=1):
            original_sentence = row.get(source_column, "")
            original_numbers = extract_numbers(original_sentence)
            if not original_numbers:
                continue

            model_output = row.get(output_column, "")
            output_numbers = extract_numbers(model_output)
            comparison = compare_numbers(original_numbers, output_numbers, model_output)

            details.append(
                {
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "row_id": index,
                    "original_numbers": ", ".join(original_numbers),
                    "output_numbers": ", ".join(output_numbers),
                    "lost_number_values": ", ".join(comparison["lost_number_values"]),
                    "has_number_loss": str(comparison["has_number_loss"]),
                    "original_sentence": original_sentence,
                    "reference_sentence": row.get(reference_column, "") if reference_column else "",
                    "model_output": model_output,
                }
            )

    return details


def build_wikismall_examples() -> list[dict[str, Any]]:
    model_order = ("Logistic Regression", "SVM", "Random Forest")
    loaded: dict[str, tuple[list[dict[str, Any]], str, str, str | None]] = {}

    for model_name in model_order:
        prediction_file, _ = find_prediction_file(model_name, "Wikismall")
        if prediction_file is None:
            print(
                f"Warning: missing Wikismall examples file for {model_name}; examples may be incomplete."
            )
            continue
        try:
            loaded[model_name] = load_rows(prediction_file)
        except Exception as exc:
            print(f"Warning: could not load Wikismall examples for {model_name}: {exc}")

    if "Logistic Regression" not in loaded:
        return []

    logistic_rows, source_column, _, reference_column = loaded["Logistic Regression"]
    max_rows = min(len(value[0]) for value in loaded.values()) if loaded else 0
    examples: list[dict[str, Any]] = []

    for index in range(max_rows):
        logistic_row = logistic_rows[index]
        original = logistic_row.get(source_column, "")
        numbers = extract_numbers(original)
        if not numbers:
            continue

        example = {
            "row_id": index + 1,
            "original_numbers": ", ".join(numbers),
            "any_model_number_loss": "False",
            "original_wikismall_input": original,
            "wikismall_simplified_sentence": logistic_row.get(reference_column, "")
            if reference_column
            else "",
            "logistic_regression_number_loss": "",
            "logistic_regression_output": "",
            "svm_number_loss": "",
            "svm_output": "",
            "random_forest_number_loss": "",
            "random_forest_output": "",
        }

        any_model_number_loss = False

        for model_name, loss_column, output_column_name in (
            (
                "Logistic Regression",
                "logistic_regression_number_loss",
                "logistic_regression_output",
            ),
            ("SVM", "svm_number_loss", "svm_output"),
            ("Random Forest", "random_forest_number_loss", "random_forest_output"),
        ):
            if model_name not in loaded:
                continue
            rows, _, output_column, _ = loaded[model_name]
            model_output = rows[index].get(output_column, "") if index < len(rows) else ""
            output_numbers = extract_numbers(model_output)
            comparison = compare_numbers(numbers, output_numbers, model_output)
            has_number_loss = comparison["has_number_loss"]
            any_model_number_loss = any_model_number_loss or has_number_loss
            example[loss_column] = str(has_number_loss)
            example[output_column_name] = model_output

        example["any_model_number_loss"] = str(any_model_number_loss)

        examples.append(example)

    return examples


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
    results: list[dict[str, Any]],
    sentence_details: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> None:
    summary_rows = [
        [
            "model_name",
            "dataset",
            "total_sentence_pairs",
            "sentences_with_numbers",
            "total_numbers_in_original",
            "preserved_numbers",
            "lost_numbers",
            "number_preservation_rate",
            "number_loss_rate",
        ],
        *[
            [
                row["model_name"],
                row["dataset"],
                row["total_sentence_pairs"],
                row["sentences_with_numbers"],
                row["total_numbers_in_original"],
                row["preserved_numbers"],
                row["lost_numbers"],
                row["number_preservation_rate"],
                row["number_loss_rate"],
            ]
            for row in results
        ],
    ]
    sentence_headers = [
        "model_name",
        "dataset",
        "row_id",
        "original_numbers",
        "output_numbers",
        "lost_number_values",
        "has_number_loss",
        "original_sentence",
        "reference_sentence",
        "model_output",
    ]
    sentence_rows = [
        sentence_headers,
        *[[row[header] for header in sentence_headers] for row in sentence_details],
    ]
    example_headers = [
        "row_id",
        "original_numbers",
        "any_model_number_loss",
        "original_wikismall_input",
        "wikismall_simplified_sentence",
        "logistic_regression_number_loss",
        "logistic_regression_output",
        "svm_number_loss",
        "svm_output",
        "random_forest_number_loss",
        "random_forest_output",
    ]
    example_rows = [example_headers] + [
        [row[header] for header in example_headers] for row in examples
    ]
    sheets = {
        "Results": (summary_rows, [24, 14, 18, 24, 24, 20, 14, 22, 14]),
        "Sentence Details": (sentence_rows, [24, 14, 10, 24, 24, 22, 16, 70, 70, 70]),
        "Wikismall Examples": (example_rows, [10, 20, 18, 70, 70, 24, 70, 16, 70, 24, 70]),
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
            "<dc:creator>Number Preservation Analysis</dc:creator>"
            "<dc:title>Number Preservation Results</dc:title>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
            "</cp:coreProperties>",
        )
        archive.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>Python</Application></Properties>",
        )
        for index, (_, (rows, widths)) in enumerate(sheets.items(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml(rows, widths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Evaluate one prediction CSV/JSON/JSONL/TSV file.")
    parser.add_argument("--model-name", default="Model", help="Label used when --input is provided.")
    parser.add_argument("--dataset-name", default="Dataset", help="Dataset label used when --input is provided.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting number preservation analysis")

    results: list[dict[str, Any]] = []
    all_checked: dict[str, list[Path]] = {}

    if args.input is not None:
        rows, source_column, output_column, _ = load_rows(args.input)
        if not rows:
            raise ValueError(f"No rows found in prediction file: {args.input}")
        results.append(
            evaluate_model(args.model_name, args.dataset_name, rows, source_column, output_column)
        )
        print("Saving results")
        save_results(results)
        print("Analysis complete")
        return

    for model_name, dataset_name in RUN_CANDIDATES:
        run_label = f"{model_name} {dataset_name}"
        print(f"Loading {run_label} predictions")
        prediction_file, checked = find_prediction_file(model_name, dataset_name)
        all_checked[run_label] = checked

        if prediction_file is None:
            print(f"Warning: missing prediction file for {run_label}; skipping.")
            continue

        try:
            rows, source_column, output_column, _ = load_rows(prediction_file)
        except Exception as exc:
            print(f"Warning: could not load {run_label} predictions from {prediction_file}: {exc}")
            continue

        print(f"  Using {prediction_file.relative_to(PROJECT_ROOT)}")
        print("Evaluating numbers")
        results.append(evaluate_model(model_name, dataset_name, rows, source_column, output_column))

    if not results:
        checked_lines = []
        for model_name, paths in all_checked.items():
            checked_lines.append(f"{model_name}:")
            checked_lines.extend(f"  - {path}" for path in paths)
        raise FileNotFoundError(
            "No usable prediction files were found. Checked these files/locations:\n"
            + "\n".join(checked_lines)
        )

    print("Saving results")
    save_results(results)
    print("Analysis complete")


if __name__ == "__main__":
    main()

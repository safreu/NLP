# mypy: ignore-errors
"""Build debug replacement-level candidate rows from classical ML outputs."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from pipeline.neural_replacement_filter.config import DEFAULT_CONFIG, OUTPUT_FILENAMES
from pipeline.neural_replacement_filter.features import extract_feature_set_a, load_spacy_model
from pipeline.neural_replacement_filter.rules import apply_hard_rules

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)

PREDICTION_COLUMN_ALIASES = {
    "complex_sentence": ["complex_sentence", "source_sentence", "source", "complex"],
    "reference_simple_sentence": [
        "reference_simple_sentence",
        "reference_sentence",
        "reference",
        "simple_sentence",
        "target_sentence",
    ],
    "model_output_sentence": [
        "model_output_sentence",
        "candidate",
        "predicted_sentence",
        "prediction",
        "model_output",
        "output",
        "output_sentence",
    ],
}


class CandidateBuildError(RuntimeError):
    """Raised for clear, user-actionable pipeline failures."""


def normalize_token(token: str, lowercase: bool = True) -> str:
    return token.lower() if lowercase else token


def text_contains_token(text: str, token: str, lowercase: bool = True) -> bool:
    target = normalize_token(token, lowercase)
    return any(normalize_token(match, lowercase) == target for match in TOKEN_RE.findall(text))


def load_prediction_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CandidateBuildError(f"Prediction data file is missing: {path}")

    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        raw = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise CandidateBuildError(f"Unsupported prediction data format: {path.suffix}")

    rename_map: dict[str, str] = {}
    columns = set(raw.columns)
    for canonical, aliases in PREDICTION_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                rename_map[alias] = canonical
                break
        else:
            raise CandidateBuildError(
                f"Prediction data is missing a '{canonical}' column. "
                f"Available columns: {list(raw.columns)}"
            )

    return raw.rename(columns=rename_map)[
        ["complex_sentence", "reference_simple_sentence", "model_output_sentence"]
    ].fillna("")


def load_replacement_dictionary(path: Path) -> tuple[dict[str, dict[str, Any]], bool]:
    if not path.exists():
        raise CandidateBuildError(f"Replacement dictionary file is missing: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("replacements"), dict):
        sample = repr(data)[:1000]
        raise CandidateBuildError(
            "Dictionary format is unclear. Expected a JSON object with a 'replacements' mapping. "
            f"Loaded sample: {sample}"
        )

    replacements = data["replacements"]
    sample_items = list(replacements.items())[:3]
    for _source, payload in sample_items:
        if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
            raise CandidateBuildError(
                "Dictionary format is unclear. Expected each replacement entry to contain "
                f"a 'candidates' list. Sample: {sample_items}"
            )

    return replacements, bool(data.get("lowercase", True))


def selected_replacement(
    replacements: dict[str, dict[str, Any]],
    token_text: str,
    lowercase: bool,
) -> tuple[str, float] | None:
    entry = replacements.get(normalize_token(token_text, lowercase))
    if not isinstance(entry, dict):
        return None

    candidates = entry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    best = candidates[0]
    if not isinstance(best, dict):
        return None

    replacement = best.get("replacement", entry.get("best_replacement"))
    if replacement is None:
        return None

    confidence = best.get("frequency")
    if confidence is None:
        total_count = float(entry.get("total_count") or 0)
        count = float(best.get("count") or 0)
        confidence = count / total_count if total_count else 0.0

    return str(replacement), float(confidence)


def create_weak_label(
    *,
    original_word: str,
    replacement_word: str,
    reference_simple_sentence: str,
    hard_rule_blocked: bool,
    hard_rule_reason: str,
    lowercase: bool,
) -> tuple[int | str, str]:
    if hard_rule_blocked:
        return 0, f"hard_rule:{hard_rule_reason}"

    replacement_in_reference = text_contains_token(
        reference_simple_sentence, replacement_word, lowercase
    )
    original_in_reference = text_contains_token(reference_simple_sentence, original_word, lowercase)

    if replacement_in_reference:
        return 1, "reference_contains_replacement"
    if original_in_reference:
        return 0, "reference_keeps_original"
    return "uncertain", "reference_match_uncertain"


def build_candidate_rows(
    *,
    predictions: pd.DataFrame,
    replacements: dict[str, dict[str, Any]],
    lowercase: bool,
    model_source: str,
) -> list[dict[str, object]]:
    nlp = load_spacy_model()
    rows: list[dict[str, object]] = []
    replacement_pos_cache: dict[str, str] = {}

    for sentence_id, row in tqdm(
        predictions.iterrows(),
        total=len(predictions),
        desc="Building replacement candidates",
    ):
        complex_sentence = str(row["complex_sentence"])
        reference_simple_sentence = str(row["reference_simple_sentence"])
        model_output_sentence = str(row["model_output_sentence"])
        doc = nlp(complex_sentence)

        seen_positions: set[int] = set()
        for token in doc:
            if token.i in seen_positions:
                continue
            selected = selected_replacement(replacements, token.text, lowercase)
            if selected is None:
                continue
            seen_positions.add(token.i)
            replacement_word, confidence = selected
            if replacement_word not in replacement_pos_cache:
                replacement_doc = nlp(replacement_word)
                replacement_pos_cache[replacement_word] = (
                    replacement_doc[0].pos_ if len(replacement_doc) else ""
                )
            features = extract_feature_set_a(
                token=token,
                replacement_word=replacement_word,
                model_confidence=confidence,
                model_source=model_source,
            )
            feature_values = features.to_dict()
            decision = apply_hard_rules(
                original_word=token.text,
                replacement_word=replacement_word,
                pos_tag=features.pos_tag,
                ner_tag=features.ner_tag,
                is_proper_noun=features.is_proper_noun,
                is_number=features.is_number,
                is_connector_token=features.is_connector,
                replacement_pos_tag=replacement_pos_cache[replacement_word],
            )
            label, label_source = create_weak_label(
                original_word=token.text,
                replacement_word=replacement_word,
                reference_simple_sentence=reference_simple_sentence,
                hard_rule_blocked=decision.blocked,
                hard_rule_reason=decision.reason,
                lowercase=lowercase,
            )

            rows.append(
                {
                    "sentence_id": int(sentence_id),
                    "complex_sentence": complex_sentence,
                    "reference_simple_sentence": reference_simple_sentence,
                    "model_output_sentence": model_output_sentence,
                    "original_word": token.text,
                    "replacement_word": replacement_word,
                    **feature_values,
                    "hard_rule_blocked": decision.blocked,
                    "hard_rule_reason": decision.reason,
                    "label": label,
                    "label_source": label_source,
                }
            )

    return rows


def write_debug_examples(candidates: pd.DataFrame, path: Path, limit: int = 40) -> None:
    lines: list[str] = []
    for _, row in candidates.head(limit).iterrows():
        lines.extend(
            [
                f"Sentence ID: {row['sentence_id']}",
                f"Complex: {row['complex_sentence']}",
                f"Reference: {row['reference_simple_sentence']}",
                f"Original -> Replacement: {row['original_word']} -> {row['replacement_word']}",
                f"Confidence: {row['model_confidence']}",
                f"POS / NER: {row['pos_tag']} / {row['ner_tag']}",
                f"Hard rule: blocked={row['hard_rule_blocked']} reason={row['hard_rule_reason']}",
                f"Weak label: {row['label']} source={row['label_source']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(rows)

    candidate_columns = [
        "sentence_id",
        "complex_sentence",
        "reference_simple_sentence",
        "model_output_sentence",
        "model_source",
        "original_word",
        "replacement_word",
        "model_confidence",
        "pos_tag",
        "ner_tag",
        "is_proper_noun",
        "is_number",
        "is_connector",
        "original_length",
        "replacement_length",
        "length_difference",
        "hard_rule_blocked",
        "hard_rule_reason",
        "label",
        "label_source",
    ]
    candidates = candidates.reindex(columns=candidate_columns)

    blocked = candidates[candidates["hard_rule_blocked"] == True]  # noqa: E712
    uncertain = candidates[candidates["label"] == "uncertain"]
    training = candidates[candidates["label"].isin([0, 1]) & (candidates["label"] != "uncertain")]
    feature_preview = candidates[
        [
            "sentence_id",
            "original_word",
            "replacement_word",
            "model_confidence",
            "model_source",
            "pos_tag",
            "ner_tag",
            "is_proper_noun",
            "is_number",
            "is_connector",
            "original_length",
            "replacement_length",
            "length_difference",
            "label",
        ]
    ]

    candidates.to_csv(output_dir / OUTPUT_FILENAMES["all_candidates"], index=False)
    blocked.to_csv(output_dir / OUTPUT_FILENAMES["blocked"], index=False)
    uncertain.to_csv(output_dir / OUTPUT_FILENAMES["uncertain"], index=False)
    training.to_csv(output_dir / OUTPUT_FILENAMES["training"], index=False)
    feature_preview.to_csv(output_dir / OUTPUT_FILENAMES["feature_preview"], index=False)

    label_counts = Counter(str(label) for label in candidates["label"].tolist())
    distribution = {
        "label_0": label_counts.get("0", 0),
        "label_1": label_counts.get("1", 0),
        "uncertain": label_counts.get("uncertain", 0),
        "blocked": int(blocked.shape[0]),
        "total_candidates": int(candidates.shape[0]),
    }
    (output_dir / OUTPUT_FILENAMES["label_distribution"]).write_text(
        json.dumps(distribution, indent=2),
        encoding="utf-8",
    )
    write_debug_examples(candidates, output_dir / OUTPUT_FILENAMES["examples"])

    LOGGER.info("Wrote %s candidates to %s", len(candidates), output_dir)
    LOGGER.info("Label distribution: %s", distribution)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--model_source", default=DEFAULT_CONFIG.model_source)
    parser.add_argument("--debug_size", type=int, default=DEFAULT_CONFIG.debug_size)
    parser.add_argument(
        "--prediction_data_path",
        type=Path,
        default=DEFAULT_CONFIG.prediction_data_path,
    )
    parser.add_argument(
        "--replacement_dictionary_path",
        type=Path,
        default=DEFAULT_CONFIG.replacement_dictionary_path,
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        predictions = load_prediction_data(args.prediction_data_path)
        if args.mode == "debug":
            predictions = predictions.head(args.debug_size)
            LOGGER.info("Debug mode: using %s prediction rows", len(predictions))
        else:
            LOGGER.info("Full mode: using %s prediction rows", len(predictions))

        replacements, lowercase = load_replacement_dictionary(args.replacement_dictionary_path)
        LOGGER.info("Loaded %s dictionary entries", len(replacements))

        rows = build_candidate_rows(
            predictions=predictions,
            replacements=replacements,
            lowercase=lowercase,
            model_source=args.model_source,
        )
        save_outputs(rows, args.output_dir)
    except CandidateBuildError as exc:
        LOGGER.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

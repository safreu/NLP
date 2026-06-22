# mypy: ignore-errors
"""Reconstruct sentence-level outputs from neural replacement decisions."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from apply_neural_filter import threshold_slug
from tqdm import tqdm

from pipeline.neural_replacement_filter.config import OUTPUT_DIR, THRESHOLDS

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "''", "'s"}
NO_SPACE_AFTER = {"(", "[", "{", "``", "$", "#"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    output = ""
    previous = ""
    for token in tokens:
        if not output:
            output = token
        elif token in NO_SPACE_BEFORE or previous in NO_SPACE_AFTER:
            output += token
        else:
            output += " " + token
        previous = token
    return output


def match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def reconstruct_sentence(group: pd.DataFrame, decision_column: str) -> tuple[str, int]:
    sentence = str(group.iloc[0]["complex_sentence"])
    tokens = tokenize(sentence)
    candidate_records = group.to_dict("records")
    cursor = 0
    replacements = 0

    for candidate in candidate_records:
        original = str(candidate["original_word"])
        replacement = str(candidate["replacement_word"])
        while cursor < len(tokens) and tokens[cursor].lower() != original.lower():
            cursor += 1
        if cursor >= len(tokens):
            continue
        if str(candidate[decision_column]) == "accept":
            tokens[cursor] = match_case(tokens[cursor], replacement)
            replacements += 1
        cursor += 1

    return detokenize(tokens), replacements


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def build_outputs(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = scored.groupby("sentence_id", sort=True)
    for sentence_id, group in tqdm(groups, desc="Reconstructing sentences"):
        hard_rules_group = group.copy()
        hard_rules_group["hard_rules_decision"] = hard_rules_group["hard_rule_blocked"].map(
            lambda value: "reject" if as_bool(value) else "accept"
        )
        hard_output, hard_final_count = reconstruct_sentence(
            hard_rules_group, "hard_rules_decision"
        )
        neural_output, neural_final_count = reconstruct_sentence(group, "final_decision")
        num_original_replacements = int(len(group))
        blocked_mask = group["hard_rule_blocked"].map(as_bool)
        num_blocked = int(blocked_mask.sum())
        num_neural_rejected = int(((~blocked_mask) & (group["neural_decision"] == "reject")).sum())

        rows.append(
            {
                "sentence_id": sentence_id,
                "complex_sentence": group.iloc[0]["complex_sentence"],
                "reference_simple_sentence": group.iloc[0]["reference_simple_sentence"],
                "original_logistic_regression_output": group.iloc[0]["model_output_sentence"],
                "hard_rules_only_output": hard_output,
                "neural_filtered_output": neural_output,
                "threshold": threshold,
                "num_original_replacements": num_original_replacements,
                "num_blocked_by_rules": num_blocked,
                "num_rejected_by_neural_filter": num_neural_rejected,
                "num_final_replacements": neural_final_count,
            }
        )
        if hard_final_count > num_original_replacements:
            LOGGER.debug("Unexpected hard-rule count for sentence_id=%s", sentence_id)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["debug"], default="debug")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    for threshold in THRESHOLDS:
        slug = threshold_slug(threshold)
        scored_path = args.output_dir / f"scored_candidates_threshold_{slug}.csv"
        if not scored_path.exists():
            LOGGER.error(
                "Scored candidate file is missing: %s. Run apply_neural_filter.py first.",
                scored_path,
            )
            return 1
        scored = pd.read_csv(scored_path)
        outputs = build_outputs(scored, threshold)
        output_path = args.output_dir / f"filtered_outputs_threshold_{slug}.csv"
        outputs.to_csv(output_path, index=False)
        LOGGER.info("Wrote %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
